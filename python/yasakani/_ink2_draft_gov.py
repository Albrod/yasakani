# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Position-aware on / off rule for drafting with the confidence cut ("auto"), and the Split-K attention of the drafter.

The cost of a plain step and of a drafting pass (draft call + verification of up to 8 rows) both grow with the position,
at different rates (the drafter's full-attention layer reads all earlier positions; the verification pass reads them for
every row). CutGovernor keeps one average of each, so at long context it compares a drafting pass at the current position
with a plain step measured many thousands of positions earlier, and the probes it makes while off cost the most exactly
there. PosCutGovernor decides with the costs AT THE CURRENT POSITION:

- costs are kept per position band (bands double in length: < 2K, 2K .. 4K, 4K .. 8K, ...; an average of the time and of
  the position of the samples in each band); the cost at position p comes from p's band, corrected along the line through
  the nearest other band (slope >= 0), or interpolated / extrapolated from the nearest bands when p's band has no sample;
- the tokens per drafting pass (the acceptance; a property of the text, not of the position) are one running average;
- while drafting is on, one plain step every `refresh_every` passes when the plain cost of the current band is missing or
  older than `stale` positions (a plain step still emits a token, so the refresh costs at most one pass's gain);
- while off, one probe pass every `gap` plain steps; the gap doubles (up to `probe_max`) after every probe that still does
  not pay and returns to `probe_every` when drafting is on again;
- on at `on_margin`, off after `off_patience` drafting passes in a row below `off_margin` (rate of drafting / rate of plain
  steps at the current position); the gap of the probes only grows after a probe clearly below (`backoff_below`);
- when the position enters another band, or goes back (a new prompt, a truncation), the gap returns to `probe_every`; while
  off, a band without a fresh drafting sample, and a position that went back (another text), are probed at once;
- the gap grows only after `backoff_after` probes in a row clearly below (`backoff_below`).

Same interface as CutGovernor (choose, update_plain, update_spec); it reads the position through `pos_fn`. It only decides
whether a pass drafts: every emitted token is still the main model's choice (spec_step), so the output does not change.
HybridGovernor uses CutGovernor below 8,192 positions and PosCutGovernor from there; the fast path of 0.3.3 chooses it
(eng.draft_gov("hyb"), with eng.draft_attn_sk(1)) through the attribute ``draft_fast`` of the engine wrapper.
"""

import math


class _Band:
    __slots__ = ("t", "p", "n", "last")

    def __init__(self):
        self.t, self.p, self.n, self.last = 0.0, 0.0, 0, -1


class PosCost:
    """Cost of one kind of pass against the position: running averages per band, read at any position."""

    def __init__(self, beta=0.2, base=2048):
        self.beta, self.base = beta, base
        self.bands = {}

    def band(self, pos):
        return 0 if pos < self.base else int(math.log2(pos / self.base)) + 1

    def add(self, pos, seconds):
        b = self.bands.setdefault(self.band(pos), _Band())
        if b.n == 0:
            b.t, b.p = float(seconds), float(pos)
        else:
            k = self.beta
            b.t = (1 - k) * b.t + k * seconds
            b.p = (1 - k) * b.p + k * pos
        b.n += 1
        b.last = pos

    def fresh(self, pos, stale):
        b = self.bands.get(self.band(pos))
        return b is not None and b.n > 0 and pos - b.last <= stale

    def at(self, pos):
        """-> estimated seconds at `pos`, or None without any sample."""
        have = sorted((k, b) for k, b in self.bands.items() if b.n > 0)
        if not have:
            return None
        k0 = self.band(pos)
        own = next((b for k, b in have if k == k0), None)
        if own is not None:
            other = min((b for k, b in have if k != k0), key=lambda b: abs(b.p - own.p), default=None)
            if other is None or abs(other.p - own.p) < 1.0:
                return own.t
            slope = max(0.0, (own.t - other.t) / (own.p - other.p))
            return max(own.t + slope * (pos - own.p), 0.25 * own.t)
        below = [b for k, b in have if b.p <= pos]
        above = [b for k, b in have if b.p > pos]
        if below and above:
            lo, hi = below[-1], above[0]
            f = (pos - lo.p) / max(hi.p - lo.p, 1.0)
            return lo.t + f * (hi.t - lo.t)
        side = below[-2:] if below else above[:2]
        if len(side) == 1:
            return side[0].t
        a, b = side
        slope = max(0.0, (b.t - a.t) / max(b.p - a.p, 1.0))
        ref = b if below else a
        return max(ref.t + slope * (pos - ref.p), 0.25 * ref.t)


class PosCutGovernor:
    def __init__(self, pos_fn, beta=0.15, on_margin=1.04, off_margin=1.0, probe_every=16, probe_max=256, refresh_every=24,
                 stale=None, probe_beta=0.5, base=2048, off_patience=3, backoff_below=0.85, backoff_after=2,
                 far_below=0.7, far_from=65536):
        self.pos_fn = pos_fn
        self.off_patience, self.backoff_below, self.backoff_after, self.below, self.weak = off_patience, backoff_below, backoff_after, 0, 0
        self.far_below, self.far_from = far_below, far_from  # a probe far below at long context: gap to probe_max at once
        self.beta, self.on_margin, self.off_margin = beta, on_margin, off_margin
        self.probe_every, self.probe_max, self.refresh_every, self.probe_beta = probe_every, probe_max, refresh_every, probe_beta
        self.stale = stale  # None: max(512, position / 8)
        self.plain = PosCost(beta, base)
        self.spec = PosCost(beta, base)
        self.tok = None
        self.n_plain = self.n_spec = 0
        self.active, self.since, self.gap, self.since_refresh = True, 0, probe_every, 0
        self._p, self._band, self._last = 0, -1, 0
        self.log = {"refresh": 0, "probe": 0, "on": 0, "off": 0}

    def _stale(self, p):
        return self.stale if self.stale is not None else max(512, p // 8)

    def gain(self, p):
        """Rate of drafting passes over the rate of plain steps at position p (None: not measured yet)."""
        tp, ts = self.plain.at(p), self.spec.at(p)
        if tp is None or ts is None or self.tok is None or ts <= 0:
            return None
        return self.tok / ts * tp

    def choose(self):
        """-> draft now (bool), is_probe."""
        p = self._p = int(self.pos_fn())
        if self.n_plain < 2:
            return False, False  # measure plain steps first (the first one may record its graph)
        band, back = self.spec.band(p), p < self._last
        self._last = p
        g = self.gain(p)
        if band != self._band or back:
            # going back inside the same band while off, with a fresh drafting sample there that clearly loses, keeps the
            # probe gap (at 131K a probe costs several plain steps, and every truncation would restart them)
            keep = (band == self._band and not self.active and g is not None and g < self.backoff_below
                    and self.spec.fresh(p, self._stale(p)))
            self._band = band
            if not keep:
                self.gap, self.weak = self.probe_every, 0
                if not self.active and (back or not self.spec.fresh(p, self._stale(p))):
                    self.since = self.gap - 1  # probe at once
        if g is None:
            return True, False  # measure a drafting pass
        if self.active:
            self.since_refresh += 1
            if self.since_refresh >= self.refresh_every and not self.plain.fresh(p, self._stale(p)):
                self.since_refresh = 0
                self.log["refresh"] += 1
                return False, False
            return True, False
        if g >= self.on_margin and self.spec.fresh(p, self._stale(p)):
            self._on()
            return True, False
        self.since += 1
        if self.since >= self.gap:
            self.since = 0
            self.log["probe"] += 1
            return True, True
        return False, False

    def _on(self):
        self.active, self.gap, self.since_refresh, self.weak = True, self.probe_every, 0, 0
        self.log["on"] += 1

    def _off(self):
        self.active, self.since, self.below = False, 0, 0
        self.log["off"] += 1

    def update_plain(self, seconds):
        self.n_plain += 1
        if self.n_plain == 1:
            return
        self.plain.add(self._p, seconds)

    def update_spec(self, tokens, seconds, probe=False):
        self.n_spec += 1
        b = self.probe_beta if probe else self.beta
        self.tok = float(tokens) if self.tok is None else (1 - b) * self.tok + b * tokens
        if self.n_spec > 1:  # the first drafting pass may record its graph
            self.spec.add(self._p, seconds)
        g = self.gain(self._p)
        if g is None:
            return
        if self.active:
            self.below = self.below + 1 if g < self.off_margin else 0
            if self.below >= self.off_patience:
                self._off()
        elif g >= self.on_margin:
            self._on()
        elif probe:
            self.weak = self.weak + 1 if g < self.backoff_below else 0
            if self.weak >= self.backoff_after:
                self.weak = 0
                self.gap = self.probe_max if (g < self.far_below and self._p >= self.far_from) else min(self.gap * 2, self.probe_max)


class HybridGovernor:
    """CutGovernor below `switch` positions, PosCutGovernor from there on.

    On an RTX 5090 the position-aware rule paid at 32K and 131K positions but lost up to 2% at 256 positions with the fast
    path, where drafting pays only a little and its probes cost more than its better cost model gains; CutGovernor with the
    Split-K drafter attention was at or above plain steps there. Plain steps update both rules (the position rule keeps its
    bands fresh); a drafting pass updates the rule that chose it."""

    def __init__(self, pos_fn, switch=8192):
        from . import _ink2_drafter_native as ydn
        self.pos_fn, self.switch = pos_fn, int(switch)
        self.cut, self.pos = ydn.CutGovernor(), PosCutGovernor(pos_fn)
        self._cur = self.cut
        self.log = self.pos.log

    @property
    def active(self):
        return self._cur.active

    @active.setter
    def active(self, v):
        self.cut.active = v
        self.pos.active = v

    def choose(self):
        p = int(self.pos_fn())
        self._cur = self.pos if p >= self.switch else self.cut
        if self._cur is self.cut:
            self.pos._p = p
        return self._cur.choose()

    def update_plain(self, seconds):
        self.cut.update_plain(seconds)
        self.pos.update_plain(seconds)

    def update_spec(self, tokens, seconds, probe=False):
        self._cur.update_spec(tokens, seconds, probe)


# ---- hooks of the engine wrapper (Ink2Native) ------------------------------------------------------------------------------
GOV_KINDS = ("cut", "pos", "hyb")


def draft_gov(self, kind=None):
    """The on / off rule of "auto" with the confidence cut: "cut" (CutGovernor, the default), "pos" (PosCutGovernor) or
    "hyb" (CutGovernor below 8,192 positions, PosCutGovernor from there).
    Returns the rule in force. A new rule starts with no measurements."""
    if kind is not None:
        if kind not in GOV_KINDS:
            raise ValueError("draft_gov must be one of %s" % (GOV_KINDS,))
        from . import _ink2_drafter_native as ydn
        self._cgov = (PosCutGovernor(self.position) if kind == "pos" else HybridGovernor(self.position) if kind == "hyb"
                      else ydn.CutGovernor())
    g = getattr(self, "_cgov", None)
    return "pos" if isinstance(g, PosCutGovernor) else "hyb" if isinstance(g, HybridGovernor) else "cut"


def draft_attn_sk(self, mode=None):
    """Split-K attention of the drafter (DLL option attn_sk): 0 = off (dattn), 1 = full-attention layers, 2 = every layer."""
    if mode is not None:
        self.draft_set("attn_sk", int(mode))
        self._d_attn_sk = int(mode)
    return getattr(self, "_d_attn_sk", 0)


def after_load(self):
    """End of draft_load: with ``self.draft_fast`` (set by the package for the fast path) the Split-K attention of the
    drafter's full-attention layers and the hybrid on / off rule; else the drafter as in 0.3.2."""
    self._d_attn_sk = 0
    if getattr(self, "draft_fast", False):
        draft_attn_sk(self, 1)
        draft_gov(self, "hyb")


def install(cls):
    for f in (draft_gov, draft_attn_sk):
        setattr(cls, f.__name__, f)
