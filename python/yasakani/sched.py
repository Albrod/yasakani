# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Several requests at once on one engine (``serve --parallel N``; experimental, 0.1.7).

The engine (the carrier) holds N stream slots (see the multi-stream notes in engine.py).  One thread, the scheduler, makes the calls
of the engine library (a generation of the carrier runs on the worker thread of :meth:`Engine.generate`, as on the single path, while
the scheduler waits for it).  The only call that comes from other threads is the cancel of the generation the carrier runs alone, made
under the scheduler's lock after it has shown that this generation is the one of the request that asks for it (or, when a request
comes, the one that is to move into a slot); the scheduler checks the same at every pass boundary of that generation, so a cancel
that the generation's start has reset is made again.

What one round of the scheduler does:

1. Clean up: streams whose requests were cancelled end at this pass boundary.
2. Admit one waiting request.  The carrier's sampling is set to the request's values (seed included: one per request), the carrier
   reads it (its prompt pass and first id) and ``ms_load`` copies it into a slot right after, with nothing in between: the slot then
   knows its frontier and start, and takes the carrier's sampling with it (no other setting follows, so the slot samples exactly as
   the carrier did).  When nothing else runs, the request runs on the carrier alone as on the single path (all of its speed), until a
   second request comes: then the carrier stops at a pass boundary and the request moves into a slot.
3. One pass of at most 16 rows: every running stream gets ``min(8, 16 // streams)`` rows (rows left over go round), its frontier and
   the drafter's drafts, never more rows than ids it may still emit.  The drafts a pass confirmed are counted as on the single path:
   while a draft equals the token the pass gave after the row before it, and a confirmed stop id ends the stream there
   (:func:`ms_accept`).  ``ms_commit`` follows every pass, also when nothing was confirmed.
4. The new ids go to the requests (each request reads them on its own thread; a slow client does not hold the others).

When one stream is left (and nothing waits), it goes back to the carrier alone (``ms_unload``, then the frontier form of the
generation call): from there it runs as fast as a single request.  The next stage of a request (after a thinking budget or a loop cut)
and the next turn of a conversation are read on the carrier after ``ms_unload`` of the slot that holds the conversation, so the carrier
reads them the way it reads them when the conversation is served alone (the same prompt-reuse mode).  So that it can, the finished
stream of a request that is still open is kept for its next stage, in its slot or on the carrier (until RESERVE_S after the request's
thread has taken the end of the stage; a next stage that came within that time keeps it until it is served).  A request whose answer
asks for the next turn at once (tool calls: the client sends the results right away) keeps it after it has closed for about as long as
reading the conversation again would take (``Scheduler.turn_keep_s``, from KEEP_TURN_MIN_S to KEEP_TURN_MAX_S, or ``keep_turn``;
``ScheduledEngine.close(next_turn=True)``; the server closes it before the end of the answer goes out).  The first new request that
comes within that time and whose prompt continues the stream is the next turn: it claims the stream when it is submitted, the stream
then stays kept for it until it is served (at most RESERVE_MAX_S after the close), and it is read before the requests that wait, as a
later stage is.  Only that request continues a claimed stream; a request that came before the answer ended is not its next turn (two
conversations with the same tokens) and does not take the stream while it is kept.  Nothing else takes a kept stream: other requests
wait for a slot meanwhile, also when nothing runs (a request runs alone on the carrier only when a slot is left for it to move into),
and a request that runs alone does not move into a kept slot (it goes on alone until a slot comes free).  But a new request does not
wait for next turns without end: the time in which only next turns stand in the way of the oldest new request is counted (the
kept-wait clock): while places are kept for them, and while next turns that came after it are read and run in slots (they came first
only because they continue kept streams).  Once a new request has waited that way as long as reading a kept conversation again would
take (``Scheduler.turn_wait_s``: that and WAIT_FLOOR_S, or ``max_wait``), it takes that place as soon as one is kept (in a slot, before
the next turns that came after it); that conversation's next turn is then read from the start.  The places that open requests keep
for their later stages are never taken.  A later stage of a request is never refused for a full queue and comes before new requests,
also when its stream was given away meanwhile (then it is read from the start): while one waits for a place, no new request is read.
When nothing runs and such a stage (or a next turn) finds no slot, it runs alone on the carrier; when it waits only because the
carrier's stream, kept for another conversation's next turn, has no place to go, that stream is given up.  Requests with images run
on the carrier alone while the others wait.  An engine failure ends every request with an error; the scheduler then stops.

The ids of a request are those it gets alone on the single path, given the same sampling, seed and prompt-reuse mode (``snap_mode`` and
``cached_tokens`` of the result say how its prompt was read).
"""
from __future__ import annotations

import itertools
import math
import queue
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._native import YasakaniError
from .engine import CONTEXT_MARGIN, MS_ROWS, Engine, GenResult, MsInfo, Sampling, ms_accept, stop_pair
from .runtime import EOS_IDS

STREAM_ROWS = 8                    # rows of one stream in a pass at most: its frontier and up to 7 drafts
RETRY_AFTER_S = 2                  # what a server tells a client that found the queue full
RESERVE_S = 2.0                    # a request that is still open keeps its finished stream this long for its next stage
                                   # (counted from the moment its thread has taken the end of the stage)
RESERVE_MAX_S = 30.0               # ... and at most this long while its thread has not taken the end (a slow client), or while its
                                   # next stage, sent within RESERVE_S, waits for its turn
SOLO_CHECK_S = 0.05               # a request alone on the carrier that found no slot to move into looks again this often

# The stream of an answer with tool calls is kept for the conversation's next turn (the tool results) about as long as reading the
# conversation again would take: what the keep saves.  READ_S: seconds to read a prompt of so many tokens from the start, one request
# at a time (measured on the RTX 5090 with the default KV format: the median of three reads of each length with the quality model,
# the slower of the two models by 0.2 to 2 %; linear between the points and past the last).
READ_S = ((0, 0.03), (468, 0.136), (996, 0.243), (2028, 0.473), (4110, 0.958), (8255, 1.98), (16533, 4.14), (33149, 8.94),
          (66416, 20.5), (132876, 51.8), (202775, 96.1))
KEEP_TURN_MIN_S = 1.0              # the keep for the next turn (auto): at least this long (a client needs time to run a quick tool) ...
KEEP_TURN_MAX_S = 30.0             # ... and at most this long
KEEP_TURN_LIMIT_S = 600.0          # a keep (or a wait) given in seconds: at most this
WAIT_FLOOR_S = 0.25                # a new request takes a place kept for a next turn once it has waited, while places were kept, as
                                   # long as reading that conversation again would take (what the keep saves) and this much more (auto)


def read_seconds(n_tokens: int) -> float:
    """About how long reading a prompt of ``n_tokens`` from the start takes (:data:`READ_S`)."""
    n = max(0, int(n_tokens))
    for (n0, s0), (n1, s1) in zip(READ_S, READ_S[1:]):
        if n <= n1:
            return s0 + (s1 - s0) * (n - n0) / (n1 - n0)
    (n0, s0), (n1, s1) = READ_S[-2], READ_S[-1]
    return s1 + (s1 - s0) * (n - n1) / (n1 - n0)


class QueueFull(YasakaniError):
    """Too many requests wait already (a server answers 503 with Retry-After)."""
    retry_after = RETRY_AFTER_S


@dataclass
class Stage:
    """One generation call of a request.  ``affinity`` = the request (its :class:`ScheduledEngine`) whose earlier stage this one
    continues; ``kind`` "visual" / "visual_end" start and end a request with image input (``visual`` = the arguments of
    :meth:`Engine.visual_input`).  ``later``: an earlier stage of the request has been served (the request is not new: this stage
    is not counted against max_queue and comes before new requests)."""
    prompt: Sequence[int]
    max_new: int
    stop_ids: Optional[Sequence[int]] = EOS_IDS
    sampling: Optional[Sampling] = None
    affinity: Optional[object] = None
    visual: Optional[dict] = None
    kind: str = "gen"
    later: bool = False


class Ticket:
    """A submitted stage.  ``q`` receives ``("ids", new_ids)`` as the stream grows, then ``("done", GenResult)`` or
    ``("error", exception)``.  ``cancel()`` may be called from any thread: the stream ends at its next pass boundary."""
    _next = itertools.count(1)

    def __init__(self, stage: Stage, sched: "Scheduler"):
        self.id = next(Ticket._next)
        self.stage = stage
        self.q: "queue.SimpleQueue" = queue.SimpleQueue()
        self.t_submit = time.perf_counter()
        self.closed = False                              # the scheduler sent "done" or "error"
        self.kw0 = 0.0                                   # the scheduler's kept-wait clock when it was submitted
        self.newcomer = False                            # picked as a new request (it may take a place kept for a next turn)
        self.resumed = False                             # picked as the next turn that claimed its stream
        self.ahead = False                               # ... before an older new request (the kept-wait clock runs while it is read)
        self._sched = sched
        self._cancel = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        if not self._cancel.is_set():
            self._cancel.set()
            self._sched._cancelled(self)


class _Held:
    """The stream of a request that is not running: the tokens in its KV, its frontier (the token it emitted last), the stage that
    made it (its ticket), when it ended, and the next turn that claimed it (the ticket of a new request, see Scheduler.submit)."""
    __slots__ = ("tokens", "front", "owner", "visual", "tid", "t_end", "claim")

    def __init__(self, tokens: List[int], front: int, owner, visual: bool = False, tid: int = 0):
        self.tokens, self.front, self.owner, self.visual, self.tid = tokens, front, owner, visual, tid
        self.t_end = time.perf_counter()
        self.claim: Optional[Ticket] = None


class _Slot:
    __slots__ = ("idx", "stream", "held", "used")

    def __init__(self, idx: int):
        self.idx = idx
        self.stream: Optional["_Stream"] = None          # running
        self.held: Optional[_Held] = None                # idle: a finished stream whose KV is kept (None and no stream: free)
        self.used = 0


class _Stream:
    """A request that runs (in a slot, or on the carrier alone)."""

    def __init__(self, ticket: Ticket, limit: int, stops: Tuple[int, ...]):
        st = ticket.stage
        self.ticket = ticket
        self.owner = st.affinity
        self.stops = stops                                  # the stop ids the single path ends at (engine.stop_pair)
        self.seq: List[int] = [int(x) for x in st.prompt]   # the prompt and the ids emitted so far (the last one is the frontier)
        self.n_prompt = len(self.seq)
        self.ids: List[int] = []
        self.limit = limit                                  # ids of this stage at most (as the single path: plan() + 1)
        self.slot: Optional[_Slot] = None
        self.snap_mode, self.cached, self.prefill_ms, self.solo_ms = 0, 0, 0.0, 0.0
        self.passes = self.ms_passes = self.streams_sum = self.drafted = self.accepted = self.moves = self.back = 0
        self.t_admit = self.t_read = 0.0
        self.kept_wait = 0.0                                # seconds it waited only because of next turns (the kept-wait clock)

    def push(self, new: List[int]) -> None:
        if new:
            self.ids += new
            self.seq += new
            self.ticket.q.put(("ids", new))

    @property
    def remaining(self) -> int:
        return self.limit - len(self.ids)

    def ended(self) -> Optional[str]:
        if self.ids and self.ids[-1] in self.stops:
            return "stop"
        if len(self.ids) >= self.limit:
            return "length"
        if self.ticket.cancelled:
            return "cancelled"
        return None

    def held(self, visual: bool = False) -> _Held:
        return _Held(self.seq[:-1], self.seq[-1], self.owner, visual, self.ticket.id)

    def read_result(self, r: GenResult, t_call: float) -> None:
        self.snap_mode, self.cached, self.prefill_ms = r.snap_mode, r.cached_tokens, r.prefill_ms
        self.t_read = t_call + r.prefill_ms / 1000.0

    def result(self, finish: str) -> GenResult:
        now = time.perf_counter()
        alone = not (self.ms_passes or self.moves)
        gen_ms = self.solo_ms if alone else 1000.0 * (now - (self.t_read or now))
        stats = dict(queue_ms=1000.0 * max(0.0, (self.t_admit or now) - self.ticket.t_submit), kept_wait_ms=1000.0 * self.kept_wait,
                     parallel_avg=(self.streams_sum / self.ms_passes) if self.ms_passes else 1.0,
                     ms_passes=float(self.ms_passes), moves=float(self.moves), back_to_one=float(self.back),
                     slot=float(self.slot.idx if self.slot is not None else -1))
        return GenResult(ids=list(self.ids), finish=finish, prompt_tokens=self.n_prompt, cached_tokens=self.cached,
                         snap_mode=self.snap_mode, passes=self.passes, prefill_ms=self.prefill_ms, gen_ms=gen_ms, stats=stats,
                         draft_n=self.drafted, draft_n_accepted=self.accepted)


def _extends(prompt: Sequence[int], tokens: List[int]) -> bool:
    """The prompt continues these tokens (the carrier would read it as a continuation of them).  The last token is looked at first:
    another conversation with the same long beginning (a system prompt) is told apart without comparing all of it (this runs in every
    round while a request waits)."""
    n = len(tokens)
    return 0 < n < len(prompt) and prompt[n - 1] == tokens[-1] and list(prompt[:n]) == tokens


def _is_open(owner) -> bool:
    return owner is not None and bool(getattr(owner, "_open", False))


def _kept(h: Optional[_Held], now: float) -> bool:
    """The finished stream of a request that is still open, kept for the request's next stage: while the request's thread has not
    taken the end of the stage (a client that reads slowly; at most RESERVE_MAX_S after the stream ended), then RESERVE_S from the moment
    the thread took it; a next stage sent within that time keeps it until the stage is served (at most RESERVE_MAX_S after the thread
    took the end).  After the request has closed, the stream of its last stage stays kept for the next turn when the request asked for
    that (:func:`_kept_closed`).  Once given up, a stream is not kept again."""
    if h is None or h.owner is None:
        return False
    return _kept_by_time(h, now) if _is_open(h.owner) else _kept_closed(h, now)


def _claim_live(h: _Held) -> Optional["Ticket"]:
    """The next turn that claimed this stream (Scheduler.submit), while it waits or runs (not once it has ended or is cancelled)."""
    c = h.claim
    return c if (c is not None and not c.closed and not c.cancelled) else None


def _kept_closed(h: _Held, now: float) -> bool:
    """:func:`_kept` for a stream whose request has closed: the stream of its last stage, when the request closed with ``keep_s`` (an
    answer with tool calls) while the stream was still kept (so that a stream given up is not kept again), until ``keep_s`` after the
    close; once its next turn has claimed it within that time, until that turn is served (at most RESERVE_MAX_S, or keep_s if longer,
    after the close)."""
    o = h.owner
    keep = float(getattr(o, "_keep_s", 0.0))
    if keep <= 0.0 or int(getattr(o, "_taken_tid", 0)) != h.tid:
        return False
    taken, closed = float(getattr(o, "_t_taken", 0.0)), float(getattr(o, "_t_closed", 0.0))
    if taken - h.t_end >= RESERVE_MAX_S or closed - taken >= RESERVE_S:
        return False
    if _claim_live(h) is not None:                           # its next turn came in time and waits for its turn
        return now - closed < max(RESERVE_MAX_S, keep)
    return now - closed < keep


def _free_for(h: _Held, st: "Stage", view: "_View") -> bool:
    """The finished stream of a closed request may be continued by stage ``st`` of another request: a stream claimed by a next turn
    only by that turn; an unclaimed one only once it is no longer kept (a request that came before the answer ended is not its next
    turn: it claimed nothing)."""
    c = _claim_live(h)
    if c is not None:
        return c.stage is st
    return not view.kept(h)


def _kept_by_time(h: _Held, now: float) -> bool:
    """:func:`_kept` for a stream whose request is open."""
    o = h.owner
    tid = int(getattr(o, "_taken_tid", 0))                   # the stage whose end the request's thread took last
    if tid < h.tid:                                          # not this one yet
        return now - h.t_end < RESERVE_MAX_S
    if tid > h.tid:                                          # a later stage ended without it (cancelled while it waited)
        return False
    taken = float(getattr(o, "_t_taken", 0.0))
    if taken - h.t_end >= RESERVE_MAX_S:                     # given up before the thread took it
        return False
    if getattr(o, "_pending", False) and float(getattr(o, "_t_sub", 0.0)) - taken < RESERVE_S:
        return now - taken < RESERVE_MAX_S                   # its next stage came in time and waits for its turn
    return now - taken < RESERVE_S


class _View:
    """What one decision of the scheduler sees of the state that the requests' threads change: one reading of the clock, and whether a
    request is open and a finished stream is kept, each read at the first look and then the same for the whole decision.  A view for a
    new request that has waited ``age`` seconds while places were kept for next turns (:meth:`aged`) takes a stream kept for a next turn
    (the stream of a closed request) as given up once ``age`` reaches ``wait(stream)``; the streams that open requests keep for their
    later stages stay kept."""
    __slots__ = ("now", "_open", "_kept", "age", "_wait")

    def __init__(self):
        self.now = time.perf_counter()
        self._open: Dict[int, bool] = {}
        self._kept: Dict[int, bool] = {}
        self.age: Optional[float] = None
        self._wait: Optional[Callable[[_Held], float]] = None

    def aged(self, age: float, wait: Callable[[_Held], float]) -> "_View":
        """The same readings, for a new request that has waited ``age`` seconds while places were kept for next turns."""
        v = _View.__new__(_View)
        v.now, v._open, v._kept, v.age, v._wait = self.now, self._open, self._kept, float(age), wait
        return v

    def is_open(self, owner) -> bool:
        if owner is None:
            return False
        v = self._open.get(id(owner))
        if v is None:
            v = self._open[id(owner)] = _is_open(owner)
        return v

    def held_kept(self, h: Optional[_Held]) -> bool:
        """Kept, whatever the age of the request this view is for."""
        if h is None or h.owner is None:
            return False
        v = self._kept.get(id(h))
        if v is None:
            v = self._kept[id(h)] = _kept_by_time(h, self.now) if self.is_open(h.owner) else _kept_closed(h, self.now)
        return v

    def kept(self, h: Optional[_Held]) -> bool:
        if not self.held_kept(h):
            return False
        if self.age is not None and not self.is_open(h.owner) and self.age >= self._wait(h):
            return False                                     # kept for a next turn, given up for a request that waited long enough
        return True


class Scheduler:
    """Serves submitted stages on ``engine`` with ``n_slots`` stream slots (made when the scheduler starts; the engine is used by the
    scheduler's thread only from then on).  ``rows``: rows of one pass at most (16).  ``max_queue``: requests that may wait beyond the
    slots a request can take now (more: :class:`QueueFull`).  ``reuse``: a finished conversation keeps its KV in its slot until the slot
    is needed (its next turn then continues it).  ``drafter``: the drafter makes the drafts of the slots.  ``back_to_one``: the last
    running stream goes back to the carrier alone.  ``keep_turn``: seconds the stream of an answer with tool calls is kept for the
    next turn (0: not kept; None: auto, :meth:`turn_keep_s`).  ``max_wait``: seconds a new request waits at most, counted while only
    next turns stand in its way (places kept for them, next turns that came after it), before it takes a kept place (None: auto,
    :meth:`turn_wait_s`).  ``trace`` (a list, for tests): what the scheduler
    did, round by round."""

    def __init__(self, engine: Engine, n_slots: int, *, rows: int = MS_ROWS, max_queue: int = 64, reuse: bool = True,
                 drafter: bool = True, back_to_one: bool = True, keep_turn: Optional[float] = None, max_wait: Optional[float] = None,
                 trace: Optional[list] = None):
        if not 1 <= int(rows) <= MS_ROWS:
            raise ValueError("rows of a pass: 1..%d" % MS_ROWS)
        if int(n_slots) < 1:
            raise ValueError("at least one stream slot")
        for name, v in (("the keep for the next turn", keep_turn), ("the longest wait for a kept place", max_wait)):
            if v is not None and not 0.0 <= float(v) <= KEEP_TURN_LIMIT_S:
                raise ValueError("%s: 0 to %g seconds" % (name, KEEP_TURN_LIMIT_S))
        self.engine = engine
        self.n_slots, self.rows, self.max_queue = int(n_slots), int(rows), int(max_queue)
        self.reuse, self.drafter, self.back_to_one = bool(reuse), bool(drafter), bool(back_to_one)
        self.keep_turn = None if keep_turn is None else float(keep_turn)
        self.max_wait = None if max_wait is None else float(max_wait)
        self.trace = trace
        self.info: Optional[MsInfo] = None
        # counters: passes, their rows and streams; seconds in the pass step (pass_s), in its engine calls (engine_s), in the rounds
        # with a pass, besides the admission of a request (round_s); reads of requests, moves into slots, returns to the carrier alone;
        # streams kept for a next turn that a new request took after waiting, or that were given up so that a waiting stage could be
        # read when nothing ran (takeovers)
        self.perf: Dict[str, float] = dict(passes=0, rows=0, streams=0, pass_s=0.0, engine_s=0.0, round_s=0.0, reads=0, moves=0,
                                           back_to_one=0, solo_runs=0, parks=0, unloads=0, takeovers=0)
        # the kept-wait clock: seconds during which the oldest new request could not be placed only because of next turns (places kept
        # for them, or next turns that came after it running in slots; it runs on while the scheduler waits or runs passes, and while
        # it reads the prompt of a next turn that came after the oldest new request, not while it reads other prompts); a request's
        # kept wait = the clock now minus the clock when it came
        self._kw_clock, self._kw_t, self._kw_on = 0.0, time.perf_counter(), False
        self._cond = threading.Condition()
        self._waiting: deque = deque()
        self._slots = [_Slot(i) for i in range(self.n_slots)]
        self._carrier: Optional[_Held] = None              # a finished stream the carrier still holds as its live sequence
        self._solo: Optional[_Stream] = None               # the stream the carrier runs alone (in engine.generate)
        self._solo_movable = False
        self._want_move = False
        self._busy: Optional[Ticket] = None                 # the ticket the scheduler works on outside the slots (admission, back)
        self._excl = None                                   # the request with image input that has the carrier to itself
        self._vis_cm = None
        self._stop = False
        self._error: Optional[BaseException] = None
        self._tick = 0
        self._rot = 0
        self._counts = (0, 0)                               # (running, idle) for status()
        self._room = self.n_slots                           # slots a new request can take (as of the last round), for submit()
        ready = threading.Event()
        self._thread = threading.Thread(target=self._main, args=(ready,), name="yasakani-scheduler", daemon=True)
        self._thread.start()
        ready.wait()
        if self._error is not None:
            self._thread.join()
            raise self._error

    # ---- the outside: submit, cancel, status, close ----------------------------------------------------------------------------
    def submit(self, stage: Stage) -> Ticket:
        """Queue a stage.  A request's later stages (``Stage.later``, or its stream is held) and the stages of a request with image
        input that has started do not count against ``max_queue``; nor do the requests that the slots free now will take (requests
        that come at once all find a place while there are slots for them).  A new request that is the next turn of a closed request
        claims the stream kept for it (:meth:`_claim_for`) and does not count either."""
        t = Ticket(stage, self)
        with self._cond:
            if self._error is not None:
                raise YasakaniError("the engine failed earlier: %s" % self._error)
            if self._stop:
                raise YasakaniError("the scheduler has stopped")
            view = _View()
            later = self._later(stage)
            claim = None if later else self._claim_for(stage, view)
            if not later and claim is None \
                    and sum(1 for w in self._waiting if not self._later(w.stage)) >= self.max_queue + max(0, self._room):
                raise QueueFull("too many requests are waiting (%d); try again in %d s" % (self.max_queue, RETRY_AFTER_S))
            if claim is not None:
                claim.claim = t                              # kept for this request until it is served (_kept_closed)
            t.kw0 = self._kw_value(time.perf_counter())
            self._waiting.append(t)
            if self._solo is not None and self._solo_movable and not self._want_move:
                self._want_move = True                       # the stream on the carrier moves into a slot at its next pass boundary
                self.engine.cancel()
            self._cond.notify_all()
        return t

    def status(self) -> dict:
        with self._cond:
            run, idle = self._counts
            return dict(parallel=self.n_slots, slots_busy=run, slots_idle=idle, queued=len(self._waiting),
                        state="error" if self._error is not None else ("stopped" if self._stop else "ok"),
                        error=None if self._error is None else str(self._error))

    def turn_keep_s(self, n_tokens: int) -> float:
        """How long the stream of an answer with tool calls (``n_tokens`` in its KV: the conversation so far) is kept for the next
        turn: ``keep_turn`` when given, else about the time reading the conversation again would take (what the keep saves), at least
        KEEP_TURN_MIN_S and at most KEEP_TURN_MAX_S."""
        if self.keep_turn is not None:
            return self.keep_turn
        return min(KEEP_TURN_MAX_S, max(KEEP_TURN_MIN_S, read_seconds(n_tokens)))

    def turn_wait_s(self, n_tokens: int) -> float:
        """How long a new request waits at most for the place of a stream kept for a next turn (``n_tokens`` in its KV), counted while
        it could not be placed only because of next turns (the kept-wait clock, :meth:`_pick`): ``max_wait`` when given, else the
        time reading that conversation again would take (what keeping it saves; at most KEEP_TURN_MAX_S) and WAIT_FLOOR_S.  Then it
        takes the place as soon as it is kept (not while a next turn runs in it), and that conversation's next turn is read from the
        start.  The places open requests keep for their later stages are not
        taken (RESERVE_S)."""
        if self.max_wait is not None:
            return self.max_wait
        return WAIT_FLOOR_S + min(KEEP_TURN_MAX_S, read_seconds(n_tokens))

    def _wait_of(self, h: _Held) -> float:
        return self.turn_wait_s(len(h.tokens) + 1)

    def _kw_value(self, now: float) -> float:
        """The kept-wait clock now (under the lock)."""
        return self._kw_clock + (now - self._kw_t if self._kw_on else 0.0)

    def _kw_set(self, now: float, on: bool) -> float:
        """Bring the kept-wait clock to ``now`` and let it run on (``on``) or not (under the lock); returns its value."""
        self._kw_clock, self._kw_t, self._kw_on = self._kw_value(now), now, bool(on)
        return self._kw_clock

    @property
    def error(self) -> Optional[BaseException]:
        return self._error

    def close(self, timeout: Optional[float] = 30.0) -> None:
        """Stop: running requests end ("cancelled" at their next pass boundary), waiting ones get an error."""
        with self._cond:
            self._stop = True
            if self._solo is not None:
                self.engine.cancel()
            self._cond.notify_all()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout)

    def _cancelled(self, t: Ticket) -> None:
        with self._cond:
            if self._solo is not None and self._solo.ticket is t:
                self.engine.cancel()                         # the carrier runs exactly this request: stop it now
            self._cond.notify_all()

    def _released(self, owner) -> None:
        with self._cond:
            self._cond.notify_all()

    def _knows(self, owner) -> bool:
        """A stream of this request is held somewhere (a later stage of a running request)."""
        if self._excl is not None and owner is self._excl:
            return True
        c = self._carrier
        return (c is not None and c.owner is owner) or any(s.held is not None and s.held.owner is owner for s in self._slots)

    def _later(self, stage: Stage) -> bool:
        """A stage of a request that has started (not a new request): not counted against max_queue, and admitted first."""
        return stage.kind == "visual_end" or stage.later or (stage.affinity is not None and self._knows(stage.affinity))

    def _claim_for(self, stage: Stage, view: "_View") -> Optional[_Held]:
        """The stream a new request claims when it is submitted, as the next turn of a closed request: a stream kept for that turn
        (:func:`_kept_closed`) that no other request has claimed and that the prompt continues (the longest; of streams with the same
        tokens, two conversations alike, the one whose request closed first: its keep runs out first, the other stays kept for the
        other conversation's turn).  A request submitted before that answer ended claims nothing later: it is not the next turn."""
        if stage.kind != "gen" or stage.later:
            return None
        best, key = None, None
        for h in [self._carrier] + [s.held for s in self._slots if s.stream is None]:
            if (h is not None and not h.visual and not view.is_open(h.owner) and _claim_live(h) is None and view.kept(h)
                    and _extends(stage.prompt, h.tokens)):
                k = (len(h.tokens), -float(getattr(h.owner, "_t_closed", 0.0)))
                if best is None or k > key:
                    best, key = h, k
        return best

    def _claimed(self, stage: Stage) -> Optional[_Held]:
        """The stream this stage claimed (:meth:`_claim_for`), while the carrier or an idle slot still holds it."""
        for h in [self._carrier] + [s.held for s in self._slots if s.stream is None]:
            if h is not None and h.claim is not None and h.claim.stage is stage:
                return h
        return None

    def _resumes(self, stage: Stage, view: "_View") -> bool:
        """A new request that is the next turn of a closed request: it claimed that request's stream when it came, and the stream is
        still kept for it.  It is admitted first, as a later stage (and was not refused for a full queue)."""
        if stage.kind != "gen" or stage.later:
            return False
        h = self._claimed(stage)
        return h is not None and view.kept(h)

    # ---- the scheduler's thread --------------------------------------------------------------------------------------------------
    def _main(self, ready: threading.Event) -> None:
        try:
            self.info = self.engine.ms_open(self.n_slots, drafter=self.drafter)
        except BaseException as ex:                          # noqa: BLE001  (reported by the constructor)
            self._error = ex
            ready.set()
            return
        ready.set()
        try:
            while self._round():
                pass
        except BaseException as ex:                          # noqa: BLE001  an engine failure ends every request
            self._fail(ex)
        finally:
            self._shutdown()

    def _log(self, *item) -> None:
        if self.trace is not None:
            self.trace.append(item)

    def _round(self) -> bool:
        with self._cond:
            while True:
                if self._stop:
                    return False
                self._drop_cancelled_waiting()
                t = self._pick()
                if t is not None or self._has_work():
                    break
                # a slot kept for a next stage is given up after RESERVE_S; a new request that waits for kept places looks again soon
                self._cond.wait(SOLO_CHECK_S if self._kw_on else (0.25 if self._waiting else 1.0))
            if t is not None:                                # the kept-wait clock does not run while a prompt is read, except that
                self._kw_set(time.perf_counter(), t.ahead)   # of a next turn read before an older new request (_pick)
        t0 = time.perf_counter()
        self._tick += 1
        self._end_cancelled()
        self._free_released()
        if t is not None:
            self._busy = t
            self._admit(t)
            self._busy = None
        t1 = time.perf_counter()
        passed = False
        if self._excl is None:
            act = [s.stream for s in self._slots if s.stream is not None]
            if act:
                if self._back_ok(act):
                    self._busy = act[0].ticket
                    self._back(act[0])
                    self._busy = None
                else:
                    self._pass(act)
                    passed = True
        self._count()
        if passed:                                           # what a round with a pass cost besides admitting a request
            self.perf["round_s"] += time.perf_counter() - (t1 if t is not None else t0)
        return True

    def _has_work(self) -> bool:
        running = any(s.stream is not None for s in self._slots)
        if self._excl is not None:                           # the others wait; a cancel still ends a stream
            return any(s.stream is not None and s.stream.ticket.cancelled for s in self._slots)
        return running

    def _drop_cancelled_waiting(self) -> None:
        for t in [w for w in self._waiting if w.cancelled and w.stage.kind == "gen"]:
            self._waiting.remove(t)
            n = len(t.stage.prompt)
            self._send(t, ("done", GenResult(ids=[], finish="cancelled", prompt_tokens=n, cached_tokens=0, snap_mode=0, passes=0,
                                             prefill_ms=0.0, gen_ms=0.0, stats=dict(queue_ms=1000.0 * (time.perf_counter() - t.t_submit)))))

    def _pick(self) -> Optional[Ticket]:
        """The stage to admit now (under the lock), or None.  First the later stages of running requests and the next turns that
        claimed their streams, in order; then the oldest new request.  A new request that cannot be placed only because of next turns
        runs the kept-wait clock: while places are kept for them (with those given up it would fit), and while next turns that came
        after it run in slots or are read before it (they came first only because they continue kept streams).  Once its kept wait
        reaches :meth:`turn_wait_s` of a kept stream, it takes that place, also before the next turns that came after it.  When
        nothing runs and a later stage or a next turn waits only because the carrier's stream, kept for another next turn, has no
        place to go, that stream is given up (:meth:`_unstick`)."""
        if not self._waiting:
            self._kw_set(time.perf_counter(), False)
            return None
        if self._excl is not None:                           # only the request with image input runs
            self._kw_set(time.perf_counter(), False)
            for t in self._waiting:
                if t.stage.affinity is self._excl:
                    self._waiting.remove(t)
                    return t
            return None
        others = len(self._waiting) > 1
        view = _View()                                       # every stage below is judged on the same state
        kw = self._kw_set(view.now, self._kw_on)             # the clock until now
        first, head = [], None                               # a running request's next stage comes first (its stream held or not),
        for t in self._waiting:                              # and the next turn of a closed request whose stream is kept for it
            if t.stage.kind == "visual_end" or self._later(t.stage):
                first.append((t, False))
            elif self._resumes(t.stage, view):
                first.append((t, True))
            elif head is None:
                head = t                                     # the oldest new request
        head_fits, aged, blocked = False, None, False
        if head is not None:
            # a new request takes no stream an open request keeps for its next stage (RESERVE_S): it waits meanwhile
            head_fits = self._fits(head.stage, others, view)
            if not head_fits and (self._turns_ahead(head)
                                  or (self._turn_kept(view) and self._fits(head.stage, others, view, math.inf))):
                blocked = True                               # only next turns stand in its way: the places kept for them, or the
                age = kw - head.kw0                          # next turns after it that run in slots
                if self._fits(head.stage, others, view, age):
                    aged = age
        self._kw_set(view.now, blocked)
        later_waits = False
        for t, resumes in first:
            if aged is not None and resumes and t.t_submit > head.t_submit:
                continue                                     # the new request waited long enough: before the next turns after it
            if t.stage.kind != "visual_end" and not self._fits(t.stage, others, view):
                later_waits = True
                continue
            self._waiting.remove(t)
            t.resumed = resumes
            t.ahead = resumes and head is not None and t.t_submit > head.t_submit
            if t.stage.kind == "gen" and not resumes and not (t.stage.affinity is not None and self._knows(t.stage.affinity)):
                self._room -= 1                              # it takes a slot (or the carrier) it did not hold
            return t
        if later_waits and self._unstick(first, others, view):
            return self._pick()                              # the carrier's stream was given up: a waiting stage fits now
        if later_waits or head is None:                      # later stages first: while one waits for a place, no new request is
            return None                                      # read.  A new one may fit where it does not (a request with images, or
                                                             # one slot): that one waits as well.  The wait is bounded: a running
                                                             # stream ends, or a kept stream is given up (RESERVE_MAX_S at most);
                                                             # when nothing runs, the later stage (or next turn) runs alone once no
                                                             # stream it would drop is kept by another request (see _plan), and a
                                                             # stream kept on the carrier for another next turn is given up
                                                             # (_unstick).
        if head_fits or aged is not None:
            self._waiting.remove(head)
            head.newcomer = True
            if head.stage.kind == "gen":
                self._room -= 1                              # it no longer waits, and takes a slot (or the carrier) until _count()
            return head
        return None

    def _turn_kept(self, view: _View) -> bool:
        """A stream is kept for a next turn (of a closed request), on the carrier or in an idle slot."""
        return any(h is not None and not view.is_open(h.owner) and view.held_kept(h)
                   for h in [self._carrier] + [s.held for s in self._slots if s.stream is None])

    def _turns_ahead(self, head: Ticket) -> bool:
        """A slot runs a next turn that came after the new request ``head``: it was read first only because it continues a kept
        stream (the kept-wait clock runs meanwhile; without it the request would wait, uncounted, as long as agents send next
        turns)."""
        return any(s.stream is not None and s.stream.ticket.resumed and s.stream.ticket.t_submit > head.t_submit for s in self._slots)

    def _unstick(self, first, others: bool, view: _View) -> bool:
        """Nothing runs, and later stages or next turns wait (``first``: none of them fits).  When the carrier holds a stream kept
        for the next turn of a closed request, and with that stream gone one of them would fit, the stream is given up (counted as a
        takeover; its turn, when it comes, is read from the start): the stages that wait need its place (each one that is read
        elsewhere must first move the carrier's stream into a slot, and no slot is left for it), so without this nothing would run
        until a keep runs out (the keep of a claimed stream lasts up to RESERVE_MAX_S).  The streams that open requests keep for
        their next stages are not given up (their wait ends within RESERVE_S)."""
        c = self._carrier
        if c is None or c.visual or view.is_open(c.owner) or not view.held_kept(c) or any(s.stream is not None for s in self._slots):
            return False
        self._carrier = None
        if not any(self._fits(t.stage, others, view) for t, _ in first):
            self._carrier = c
            return False
        self._took(None, -1)                                 # the engine's live sequence is dropped by the next read (it is not
        self._log("unstick", c.tid)                          # the carrier's stream any more)
        return True

    def _fits(self, st: Stage, others_wait: bool, view: Optional[_View] = None, age: Optional[float] = None) -> bool:
        """Whether the stage can be admitted now (see :meth:`_plan`; ``age``: a new request's kept wait)."""
        if st.kind == "gen":
            return self._plan(st, others_wait, view, age) is not None
        if st.kind == "visual":                              # setting image input drops the carrier's stream: a kept one goes into a slot
            view = view or _View()
            if age is not None:
                view = view.aged(age, self._wait_of)
            return not view.kept(self._carrier) or bool(self._spares((), view))
        return True

    def _plan(self, st: Stage, others_wait: bool, view: Optional[_View] = None, age: Optional[float] = None):
        """Where a stage of kind "gen" is read and where it runs (:meth:`_plan_in`).  ``age`` (a new request): when it finds no place,
        the seconds it has waited while places were kept for next turns; a stream kept for a next turn for which it has waited long
        enough (:meth:`turn_wait_s`) is then given up for it, and it runs in a slot (not alone: the carrier stays free for the next
        turns that wait)."""
        view = view or _View()
        p = self._plan_in(st, others_wait, view, True)
        if p is None and age is not None:
            p = self._plan_in(st, others_wait, view.aged(age, self._wait_of), False)
        return p

    def _plan_in(self, st: Stage, others_wait: bool, view: _View, alone: bool):
        """Where a stage of kind "gen" is read and where it runs, without taking a stream that another open request keeps: (on_carrier,
        src, dest), or None (it waits).  ``on_carrier``: the carrier continues the stream it holds; ``src``: the slot whose stream comes
        back to the carrier for the read (the request's own, or a finished conversation the prompt continues); ``dest``: the slot the
        stream runs in, None: on the carrier alone.  A stream the carrier holds goes into a slot before another request's read drops it;
        when its request keeps it, a slot must be left for it.  A request alone on the carrier moves into a slot when a second one comes,
        so a new request runs alone only when a slot is left for that too (its own slot ``src`` frees one; one slot: it never moves).
        A later stage (or a next turn that claimed its stream) that finds no such slot runs alone when nothing else runs (it then
        stays on the carrier until a slot comes free).
        ``view``: what the requests' threads change, as one decision sees it (:class:`_View`).  ``alone``: False: only in a slot."""
        owner = st.affinity
        c = self._carrier
        mine = self._claimed(st)                             # a next turn: the stream it claimed
        if mine is not None:
            on_carrier = mine is c
            src = None if on_carrier else next((s for s in self._slots if s.stream is None and s.held is mine), None)
        else:
            on_carrier = c is not None and ((owner is not None and c.owner is owner)
                                            or (not c.visual and not view.is_open(c.owner) and _free_for(c, st, view)
                                                and _extends(st.prompt, c.tokens)))
            src = None
            if not on_carrier:
                src = self._slot_of(owner) or (None if self._excl is not None else self._continued(st, owner, view))
        if self._excl is not None:
            return on_carrier, src, None
        spares = self._spares(() if src is None else (src.idx,), view)
        park = 1 if (not on_carrier and view.kept(c)) else 0
        running = any(s.stream is not None for s in self._slots)
        if alone and not running and (not others_wait or self.n_slots == 1):
            move = 0 if (src is not None or self.n_slots == 1) else 1
            if len(spares) >= park + move:
                return on_carrier, src, None
        else:
            dest = src if src is not None else (spares[0] if spares else None)
            if dest is not None and len(spares) - (0 if dest is src else 1) >= park:
                return on_carrier, src, dest
        if alone and not running and (self._later(st) or (mine is not None and view.kept(mine))) and len(spares) >= park:
            return on_carrier, src, None
        return None

    def _send(self, t: Ticket, msg) -> None:
        if not t.closed:
            if msg[0] in ("done", "error"):
                t.closed = True
            t.q.put(msg)

    def _count(self) -> None:
        now = time.perf_counter()
        run = sum(1 for s in self._slots if s.stream is not None) + (1 if self._solo is not None else 0)
        idle = sum(1 for s in self._slots if s.stream is None and s.held is not None)
        room = sum(1 for s in self._slots if s.stream is None and not self._reserved(s, now)) - (1 if self._solo is not None else 0)
        with self._cond:
            self._counts = (run, idle)
            self._room = room                                # the stream the carrier runs alone takes a slot when another comes

    # ---- slots ---------------------------------------------------------------------------------------------------------------
    def _end_cancelled(self) -> None:
        for s in self._slots:
            if s.stream is not None and s.stream.ticket.cancelled:
                self._end(s.stream, "cancelled")

    def _free_released(self) -> None:
        """Without reuse, a finished stream stays in its slot only while its request may still need it (a later stage)."""
        if self.reuse:
            return
        now = time.perf_counter()
        for s in self._slots:
            if s.stream is None and s.held is not None and not _is_open(s.held.owner) and not _kept(s.held, now):
                s.held = None

    def _slot_of(self, owner) -> Optional[_Slot]:
        if owner is None:
            return None
        for s in self._slots:
            if s.stream is None and s.held is not None and s.held.owner is owner:
                return s
        return None

    def _continued(self, st: Stage, owner, view: "_View") -> Optional[_Slot]:
        """The idle slot whose stream the stage's prompt continues (the longest), not one a running request keeps for itself, nor one
        kept for another request's next turn (:func:`_free_for`)."""
        best = None
        for s in self._slots:
            h = s.held
            if s.stream is None and h is not None and not h.visual \
                    and (h.owner is owner or (not view.is_open(h.owner) and _free_for(h, st, view))) \
                    and _extends(st.prompt, h.tokens) and (best is None or len(h.tokens) > len(best.held.tokens)):
                best = s
        return best

    @staticmethod
    def _reserved(s: _Slot, now: float) -> bool:
        """The slot holds the finished stream of a request that is still open and keeps it (:func:`_kept`): the request's next stage
        (or the next turn it is about to send) continues it there."""
        return s.stream is None and _kept(s.held, now)

    def _spares(self, exclude=(), view: Optional[_View] = None) -> List[_Slot]:
        """The slots another stream may fill, best first: the free ones, then the idle ones used longest ago, first those of closed
        requests, then those of open requests that no longer keep them.  Never a slot kept for an open request's next stage (or for the
        next turn of a closed request, :func:`_kept_closed`), except, last, the slots kept for next turns that the view gives up (a
        new request that waited long enough, :meth:`_View.aged`): unclaimed first, then the shortest."""
        view = view or _View()
        cand = [s for s in self._slots if s.stream is None and s.idx not in exclude]
        free = [s for s in cand if s.held is None]
        idle = sorted((s for s in cand if s.held is not None and not view.is_open(s.held.owner) and not view.held_kept(s.held)),
                      key=lambda s: s.used)
        late = sorted((s for s in cand if s.held is not None and view.is_open(s.held.owner) and not view.kept(s.held)),
                      key=lambda s: s.used)
        if view.age is None:
            return free + idle + late
        given = sorted((s for s in cand if s.held is not None and not view.is_open(s.held.owner) and view.held_kept(s.held)
                        and not view.kept(s.held)), key=lambda s: (_claim_live(s.held) is not None, len(s.held.tokens), s.used))
        return free + idle + late + given

    def _spare(self, exclude=()) -> Optional[_Slot]:
        """The slot to fill (see :meth:`_spares`), or None."""
        sp = self._spares(exclude)
        return sp[0] if sp else None

    def _park(self, exclude=(), view: Optional[_View] = None) -> None:
        """The carrier's finished stream goes into a slot before anything drops the carrier's live sequence (or is forgotten).  The
        carrier keeps showing the stream until the slot holds it (one step under the lock): a next turn submitted while it moves
        still finds it and claims it (:meth:`_claim_for`).  ``view``: that of a new request that has waited for kept places
        (:meth:`_View.aged`), when this is done for it: a stream it does not give up (the stream an open request keeps for its next
        stage, or one kept for a next turn whose wait it has not reached) may then go into a place kept for a next turn that it gives
        up (counted as taken), as its plan counted (:meth:`_plan_in`).  Without that place the stream would be dropped."""
        c = self._carrier
        if c is None:
            return
        v = view or _View()
        s = None
        if not c.visual and (self.reuse or v.is_open(c.owner) or v.held_kept(c)):
            sp = self._spares(exclude, v if v.kept(c) else None)
            s = sp[0] if sp else None
        if s is not None and self._given_up(s.held, None):
            self._took(self._busy, s.idx)                    # a place kept for a next turn, given up for a request that waited
        if s is not None:
            self.engine.ms_load(s.idx)                       # right after the generation that left it (nothing in between)
        elif self._given_up(c, None):
            self._took(self._busy, -1)                       # kept for a next turn, given up for a request that waited (_plan)
        with self._cond:
            self._carrier = None
            if s is not None:
                s.held, s.used = c, self._tick
        if s is not None:
            self.perf["parks"] += 1
            self._log("park", s.idx)

    # ---- admission -----------------------------------------------------------------------------------------------------------
    def _admit(self, t: Ticket) -> None:
        st = t.stage
        if st.kind == "visual":
            return self._visual_begin(t)
        if st.kind == "visual_end":
            return self._visual_end(t)
        try:
            limit = self.engine.plan(len(st.prompt), st.max_new) + 1
        except ValueError as ex:                             # as on the single path: the prompt leaves no room
            return self._send(t, ("error", ex))
        with self._cond:
            others = bool(self._waiting)
            kept_wait = max(0.0, self._kw_clock - t.kw0)    # the clock does not run during the admission of a new request
            # under the lock: a stream claimed meanwhile (submit) is seen
            plan = self._plan(st, others, None, kept_wait if t.newcomer else None)
        pview = _View().aged(kept_wait, self._wait_of) if t.newcomer else None   # where the carrier's stream may go (_park)
        if plan is None:                                     # not after _pick (slots only come free meanwhile): it waits on
            with self._cond:
                self._waiting.appendleft(t)
            return None
        on_carrier, src, dest = plan
        alone = dest is None
        x = _Stream(t, limit, stop_pair(st.stop_ids, self.engine.spec))
        x.t_admit = time.perf_counter()
        x.kept_wait = kept_wait
        cont = self._carrier if on_carrier else (src.held if src is not None else None)
        if self._given_up(cont, t):                          # it continues a stream kept for another request's next turn
            self._took(t, -1 if src is None else src.idx)
        if not on_carrier and self._carrier is not None:
            self._park(exclude=tuple(s.idx for s in (src, dest) if s is not None), view=pview)
        if src is not None:                                  # the request's stream comes back to the carrier: read as alone
            self.engine.ms_unload(src.idx, src.held.tokens)
            src.held = None
            self.perf["unloads"] += 1
        elif not on_carrier:
            self.engine.live_drop()                          # never continue the stream of another request by chance
        self._carrier = None
        self.engine.set_sampling(st.sampling or Sampling())
        self.perf["reads"] += 1
        self._log("read", t.id, -1 if src is None else src.idx, on_carrier, alone, -1 if dest is None else dest.idx)
        if alone:
            return self._run_alone(x, x.seq, st.max_new, frontier=False, first=True)
        t_call = time.perf_counter()
        r = self.engine.generate(x.seq, 1, stop_ids=st.stop_ids)
        x.read_result(r, t_call)
        x.passes += r.passes
        x.push([int(v) for v in r.ids])
        end = x.ended()
        if end is not None:
            return self._end(x, end, pview)
        if dest is not src and self._given_up(dest.held, t):
            self._took(t, dest.idx)                          # a stream kept for a next turn, given up for a request that waited
        self.engine.ms_load(dest.idx)                        # right after the read
        self._place(x, dest)

    @staticmethod
    def _given_up(h: Optional[_Held], t: Optional[Ticket]) -> bool:
        """``h`` is kept for the next turn of a closed request, and ``t`` is not that turn (it takes it after its wait, _plan)."""
        return (h is not None and not h.visual and h.owner is not None and not _is_open(h.owner)
                and (t is None or _claim_live(h) is not t) and _kept(h, time.perf_counter()))

    def _took(self, t: Optional[Ticket], where: int) -> None:
        self.perf["takeovers"] += 1
        self._log("takeover", -1 if t is None else t.id, where)

    def _place(self, x: _Stream, s: _Slot) -> None:
        s.stream, s.held, s.used = x, None, self._tick
        x.slot = s

    def _end(self, x: _Stream, finish: str, view: Optional[_View] = None) -> None:
        """``view``: that of the new request whose read just ended (its first id ended it), when it had waited (:meth:`_park`)."""
        if x.slot is not None:                               # the slot keeps the stream's KV
            s = x.slot
            s.stream, s.used = None, self._tick
            s.held = x.held() if (self.reuse or _is_open(x.owner)) else None
        else:                                                # the carrier holds it as its live sequence
            self._carrier = x.held(visual=self._excl is not None)
            if self._excl is None and any(s.stream is not None for s in self._slots):
                self._park(view=view)                        # the next pass would drop it
        self._send(x.ticket, ("done", x.result(finish)))

    # ---- the carrier alone ---------------------------------------------------------------------------------------------------
    def _run_alone(self, x: _Stream, prompt: List[int], max_new: int, *, frontier: bool, first: bool) -> None:
        """The carrier generates the stream on the single path until it ends or another request comes: then the stream moves into a
        slot at a pass boundary, never into one kept for an open request's next stage.  While no slot is left for it, it goes on alone
        (the requests that come wait) and looks again every SOLO_CHECK_S while a request waits."""
        movable = self._excl is None and self.n_slots >= 2
        self.perf["solo_runs"] += 1
        while True:
            seen = [0]
            look = [time.perf_counter() + SOLO_CHECK_S]

            def hook(ids):
                n = len(ids)
                if n > seen[0]:
                    x.push([int(v) for v in ids[seen[0]:n]])
                    seen[0] = n
                with self._cond:
                    if movable and not self._solo_movable and self._waiting and time.perf_counter() >= look[0]:
                        look[0] = time.perf_counter() + SOLO_CHECK_S
                        if self._spare() is not None:        # a kept slot was given up (or its request ended): move now
                            self._solo_movable = self._want_move = True
                    if x.ticket.cancelled or self._want_move or self._stop:
                        self.engine.cancel()

            with self._cond:
                self._solo = x
                self._solo_movable = movable and self._spare() is not None
                self._want_move = self._solo_movable and bool(self._waiting)
            self._count()
            t_call = time.perf_counter()
            try:
                r = self.engine.generate(prompt, max_new, stop_ids=x.ticket.stage.stop_ids, on_tokens=hook, frontier=frontier)
            finally:
                with self._cond:
                    self._solo = None
                    self._solo_movable = self._want_move = False
            if len(r.ids) > seen[0]:
                x.push([int(v) for v in r.ids[seen[0]:]])
            if first:
                x.read_result(r, t_call)
            x.passes += r.passes
            x.solo_ms += r.gen_ms
            x.drafted += r.draft_n
            x.accepted += r.draft_n_accepted
            end = x.ended()
            if end is None and self._stop:
                end = "cancelled"
            if end is not None:
                return self._end(x, end)
            s = self._spare()                                # there since the move was asked for (a slot given up stays so)
            if s is not None:
                break
            if x.remaining < 2:                              # cannot happen: the frontier form below needs two ids to go
                raise YasakaniError("scheduler: no stream slot for the request that runs alone")
            self._log("stay", x.ticket.id, len(x.ids))       # every slot is kept: it goes on alone (the frontier form, as _back)
            prompt, max_new, frontier, first = list(x.seq), x.remaining, True, False
        self.engine.ms_load(s.idx)                           # right after the generation
        self._place(x, s)
        x.moves += 1
        self.perf["moves"] += 1
        self._log("move", x.ticket.id, s.idx, len(x.ids))

    def _back_ok(self, act: List[_Stream]) -> bool:
        if not self.back_to_one or len(act) != 1:
            return False
        x = act[0]
        with self._cond:
            if self._waiting and self.n_slots > 1:
                return False
        return (not x.ticket.cancelled and x.remaining >= 2
                and self.engine.max_ctx - len(x.seq) - CONTEXT_MARGIN >= 1)

    def _back(self, x: _Stream) -> None:
        """The last running stream goes back to the carrier alone."""
        s = x.slot
        self.engine.ms_unload(s.idx, x.seq[:-1])
        s.stream = s.held = None                             # moved out: free until ms_load fills it again
        x.slot = None
        x.back += 1
        self.perf["back_to_one"] += 1
        self.perf["unloads"] += 1
        self._log("back", x.ticket.id, s.idx, len(x.ids))
        self._run_alone(x, list(x.seq), x.remaining, frontier=True, first=False)

    # ---- one pass of the slots -----------------------------------------------------------------------------------------------
    def _pass(self, act: List[_Stream]) -> None:
        t0 = time.perf_counter()
        if len(act) > self.rows:                             # more streams than rows: they take turns
            k = self._rot % len(act)
            act = (act[k:] + act[:k])[:self.rows]
        na = len(act)
        per = min(STREAM_ROWS, self.rows // na)
        extra = self.rows - per * na if per < STREAM_ROWS else 0
        drafter = self.engine.ms_drafter
        slots: List[int] = []
        toks: List[int] = []
        dev: List[int] = []
        groups = []
        for q, x in enumerate(act):
            share = per + (1 if (q + self._rot) % na < extra else 0)
            want = max(0, min(share - 1, x.remaining - 1)) if drafter else 0
            groups.append((x, len(slots), 1 + want, share))
            slots += [x.slot.idx] * (1 + want)
            toks += [x.seq[-1]] + [0] * want
            dev += [1 if want else 0] + [0] * want
        self._rot += 1
        t1 = time.perf_counter()
        r = self.engine.ms_pass(slots, toks, dev if any(dev) else None)
        t2 = time.perf_counter()
        cs: List[int] = []
        ca: List[int] = []
        done = []
        for x, r0, n, share in groups:
            a, new = ms_accept(r.toks[r0:r0 + n], r.argmax[r0:r0 + n], x.stops)
            if a:
                cs.append(x.slot.idx)
                ca.append(a)
            done.append((x, a, new))
            if self.trace is not None:
                self._log("rows", x.ticket.id, r.toks[r0:r0 + n], r.argmax[r0:r0 + n], a, new, share, x.remaining)
        t3 = time.perf_counter()
        self.engine.ms_commit(cs, ca)                        # after every pass, also with nothing confirmed
        t4 = time.perf_counter()
        for (x, a, new), (_, _, n, _) in zip(done, groups):
            x.push(new)
            x.ms_passes += 1
            x.passes += 1
            x.streams_sum += na
            x.drafted += n - 1
            x.accepted += a
            end = x.ended()
            if end is not None and end != "cancelled":       # a cancel ends the stream at the start of the next round
                self._end(x, end)
        p = self.perf
        p["passes"] += 1
        p["rows"] += len(slots)
        p["streams"] += na
        p["engine_s"] += (t2 - t1) + (t4 - t3)
        p["pass_s"] += time.perf_counter() - t0

    # ---- image input ---------------------------------------------------------------------------------------------------------
    def _visual_begin(self, t: Ticket) -> None:
        view = None
        if t.newcomer:                                       # it may have waited for kept places: where the carrier's stream goes
            with self._cond:
                view = _View().aged(max(0.0, self._kw_clock - t.kw0), self._wait_of)
        self._park(view=view)                                # setting image input resets the carrier
        self.engine.ms_flush()
        cm = self.engine.visual_input(**t.stage.visual)
        try:
            cm.__enter__()
        except (ValueError, YasakaniError) as ex:            # the request's arrays, or a library without image input
            return self._send(t, ("error", ex))
        self._vis_cm, self._excl = cm, t.stage.affinity
        self._log("visual", t.id)
        self._send(t, ("done", None))

    def _visual_end(self, t: Ticket) -> None:
        cm, self._vis_cm, self._excl, self._carrier = self._vis_cm, None, None, None
        if cm is not None:
            cm.__exit__(None, None, None)                    # clears the image input (a reset of the carrier)
        self._log("visual_end", t.id)
        self._send(t, ("done", None))

    # ---- failure and stop ----------------------------------------------------------------------------------------------------
    def _all_tickets(self) -> List[Ticket]:
        ts = [s.stream.ticket for s in self._slots if s.stream is not None]
        if self._busy is not None:
            ts.append(self._busy)
        with self._cond:
            ts += list(self._waiting)
            self._waiting.clear()
        return ts

    def _fail(self, ex: BaseException) -> None:
        err = ex if isinstance(ex, YasakaniError) else YasakaniError("scheduler failure: %r" % (ex,))
        with self._cond:
            self._error = err
            self._stop = True
            self._cond.notify_all()
        for t in self._all_tickets():
            self._send(t, ("error", YasakaniError("the engine failed: %s" % err)))

    def _shutdown(self) -> None:
        for s in self._slots:
            if s.stream is not None:
                self._send(s.stream.ticket, ("done", s.stream.result("cancelled")))
                s.stream = None
        for t in self._all_tickets():
            self._send(t, ("error", YasakaniError("the server is stopping")))
        cm, self._vis_cm = self._vis_cm, None
        if cm is not None and self._error is None:
            try:
                cm.__exit__(None, None, None)
            except Exception:                                # noqa: BLE001  stopping anyway
                pass
        self._count()


class ScheduledEngine:
    """One request's view of a :class:`Scheduler`, with the members ChatRunner uses of an :class:`Engine`: ``generate``, ``cancel``,
    ``set_sampling``, ``max_ctx``, ``supports_visual``, ``visual_input``.  Make one per request and ``close()`` it when the request is
    done (its stream may wait in a slot for the request's next stage until then).  ``client_gone()`` (optional) is asked about once a
    second while the request waits or runs; True cancels it."""

    def __init__(self, sched: Scheduler, client_gone: Optional[Callable[[], bool]] = None, *, poll_s: float = 0.25,
                 gone_every_s: float = 1.0):
        self._sched = sched
        self.engine = sched.engine
        self.client_gone = client_gone
        self.sampling: Optional[Sampling] = None
        self.results: List[GenResult] = []
        self._poll_s, self._gone_every = float(poll_s), float(gone_every_s)
        self._ticket: Optional[Ticket] = None
        self._open = True
        self._served = False                                 # a stage of this request has been served (the next ones are later)
        self._pending = False                                # a stage is submitted and its end not taken yet by this thread
        self._t_sub = 0.0                                    # when this thread submitted its last stage (perf_counter)
        self._t_taken = 0.0                                  # when this thread took the end of its last stage (perf_counter)
        self._taken_tid = 0                                  # ... and the ticket of that stage
        self._keep_s = 0.0                                   # close(keep_s): its last stream is kept this long for the next turn
        self._t_closed = 0.0                                 # when it closed (perf_counter)
        self._last_n = 0                                     # the tokens of its last stream (the prompt and the ids of its last stage)

    @property
    def max_ctx(self) -> int:
        return self.engine.max_ctx

    @property
    def supports_visual(self) -> bool:
        return self.engine.supports_visual

    def plan(self, n_prompt: int, max_new: int) -> int:
        return self.engine.plan(n_prompt, max_new)

    def set_sampling(self, s: Sampling) -> Sampling:
        """The request's sampling (with its seed), used by all of its stages."""
        self.sampling = s.normalized()
        return self.sampling

    def cancel(self) -> None:
        t = self._ticket
        if t is not None:
            t.cancel()

    def close(self, keep_s: float = 0.0, *, next_turn: bool = False) -> None:
        """The request is done (close it before its answer goes out: its next turn may come at once).  ``next_turn`` (an answer whose
        next turn comes at once: tool calls): the stream of its last stage stays kept for the next turn, a new request whose prompt
        continues it (read first, as a later stage is), as long as the scheduler keeps such a stream of its length
        (:meth:`Scheduler.turn_keep_s`); ``keep_s`` > 0 gives that time instead.  The first such request claims the stream, which then
        stays kept until that request is served."""
        if self._open:
            if next_turn and keep_s <= 0.0 and self._last_n > 0:
                keep_s = self._sched.turn_keep_s(self._last_n)
            self._keep_s = max(0.0, float(keep_s))           # before _open: the scheduler never sees it closed without these
            self._t_closed = time.perf_counter()
            self._open = False
            self._sched._released(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _submit(self, st: Stage) -> Ticket:
        st.later = self._served
        self._t_sub = time.perf_counter()
        self._pending = True                                 # before the scheduler can end the stage: its slot is kept until taken
        try:
            return self._sched.submit(st)
        except BaseException:
            self._pending = False
            raise

    def _wait(self, t: Ticket, on_tokens=None, limit: int = 0):
        buf = np.empty(max(1, limit), np.int32)
        n = 0
        failure: Optional[BaseException] = None
        next_check = time.monotonic() + self._gone_every
        self._ticket = t
        try:
            while True:
                try:
                    msgs = [t.q.get(timeout=self._poll_s)]
                except queue.Empty:
                    msgs = []
                while True:
                    try:
                        msgs.append(t.q.get_nowait())
                    except queue.Empty:
                        break
                grew = False
                result, error = None, None
                for kind, val in msgs:
                    if kind == "ids":
                        buf[n:n + len(val)] = val
                        n += len(val)
                        grew = True
                    elif kind == "done":
                        result = (val,)
                    else:
                        error = val
                if grew and on_tokens is not None and failure is None:
                    try:
                        on_tokens(buf[:n])
                    except BaseException as ex:              # noqa: BLE001  (a consumer that fails stops the request)
                        failure = ex
                        t.cancel()
                if error is not None:
                    raise error
                if result is not None:
                    break
                if self.client_gone is not None and time.monotonic() >= next_check:
                    next_check = time.monotonic() + self._gone_every
                    if self.client_gone():
                        t.cancel()
        except BaseException:
            t.cancel()                                       # e.g. Ctrl+C: the stream ends at its next pass boundary
            raise
        finally:
            self._ticket = None
            self._t_taken = time.perf_counter()              # RESERVE_S counts from here (the last ids were delivered)
            self._taken_tid = t.id
            self._pending = False
        self._served = True
        r = result[0]
        if isinstance(r, GenResult):
            self._last_n = int(r.prompt_tokens) + len(r.ids)
        if failure is not None:
            raise failure
        return r

    def generate(self, prompt: Sequence[int], max_new: int, *, stop_ids: Optional[Sequence[int]] = EOS_IDS,
                 on_tokens: Optional[Callable[[np.ndarray], None]] = None, spec: Optional[Dict[str, int]] = None,
                 poll_s: Optional[float] = None, frontier: bool = False) -> GenResult:
        """As :meth:`Engine.generate`.  The request's later stages (a prompt that continues an earlier stage of this request) continue
        the request's stream."""
        if spec:
            raise ValueError("per-request speculation settings are not used when requests are served in parallel")
        if frontier:
            raise ValueError("the frontier form is the scheduler's own")
        p = [int(x) for x in np.asarray(prompt, dtype=np.int64).ravel()]
        limit = self.engine.plan(len(p), max_new) + 1          # raises as on the single path
        st = Stage(p, int(max_new), None if stop_ids is None else tuple(int(x) for x in stop_ids),
                   self.sampling or Sampling().normalized(), self)
        r = self._wait(self._submit(st), on_tokens, limit)
        self.results.append(r)
        return r

    @contextmanager
    def visual_input(self, features, slots, positions, delta):
        """Image input for the stages inside the block: the request has the carrier to itself until the block ends (the other requests
        wait).  The arrays are checked by :meth:`Engine.visual_input`."""
        if not self.supports_visual:
            raise YasakaniError("the engine library has no yskn_set_visual export")
        self._wait(self._submit(Stage([], 0, None, None, self, dict(features=features, slots=slots, positions=positions, delta=delta),
                                      "visual")))
        try:
            yield
        finally:
            self._wait(self._submit(Stage([], 0, None, None, self, None, "visual_end")))


# ---- the start-up check of serve --parallel ------------------------------------------------------------------------------------------
CHECK_TEXTS = (
    "The history of the printing press began",
    "def merge_intervals(intervals):",
    "Photosynthesis turns light into",
    "Once upon a time, a lighthouse keeper",
    "Why is the square root of two irrational? Because",
    "A hash table stores its keys in",
    "{\"name\": \"Alice\", \"age\": 30, \"city\":",
    "SELECT name, COUNT(*) FROM orders",
    "Rivers carry sediment toward the sea, where",
    "In mathematics, a prime number is",
    "Three eggs, flour and",
    "Dear team, thank you for",
    "#include <stdio.h>\nint main(void) {",
    "Mount Fuji is the highest mountain in",
    "To change a flat tire, first",
    "Orchestras tune their instruments while",
)
CHECK_NEW = 96                     # ids each request of the check generates (stop ids ignored: every request runs as long)
CHECK_SAMPLINGS = (Sampling(temperature=0.0), Sampling(1.0, 0.95, 20, 0.0, 7, 3), Sampling(0.6, 0.9, 20, 0.05, 8, 3))


def check_prompts(fmt, n: int) -> List[List[int]]:
    """``n`` short plain prompts (no chat template) for :func:`consistency_check`: the first 16 begin with different words, and no
    prompt is the start of another."""
    out = []
    for i in range(int(n)):
        text = CHECK_TEXTS[i % len(CHECK_TEXTS)]
        if i >= len(CHECK_TEXTS):
            text = "(%d) %s" % (i // len(CHECK_TEXTS) + 1, text)
        out.append([int(x) for x in fmt.encode(text)])
    return out


@dataclass
class CheckResult:
    """What :func:`consistency_check` found.  ``compared``: requests whose prompt was read the same way both times (the same
    ``snap_mode`` and ``cached_tokens``: only then are their ids the same by the engine's contract); ``differ``: of these, the ones
    whose ids or finish differ, with the first position that differs in ``first_diff``."""
    n: int
    compared: int
    differ: List[int] = field(default_factory=list)
    first_diff: Dict[int, int] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.differ

    def line(self) -> str:
        if self.ok:
            return ("the same ids for %d of %d requests (%d read another way, not compared) in %.0f s"
                    % (self.compared, self.n, self.n - self.compared, self.seconds))
        return ("different ids for %d of %d requests (%s) in %.0f s"
                % (len(self.differ), self.compared, ", ".join("#%d from id %d" % (i + 1, self.first_diff[i] + 1) for i in self.differ),
                   self.seconds))


def compare_runs(alone: Sequence[GenResult], together: Sequence[GenResult]) -> CheckResult:
    """Request by request: the ids and finish of the run alone against the run with the others."""
    res = CheckResult(n=len(alone), compared=0)
    for i, (a, b) in enumerate(zip(alone, together)):
        if (a.snap_mode, a.cached_tokens) != (b.snap_mode, b.cached_tokens):
            continue
        res.compared += 1
        if list(a.ids) != list(b.ids) or a.finish != b.finish:
            res.differ.append(i)
            res.first_diff[i] = next((k for k, (x, y) in enumerate(zip(a.ids, b.ids)) if x != y), min(len(a.ids), len(b.ids)))
    return res


def consistency_check(sched: Scheduler, prompts: Sequence[Sequence[int]], *, max_new: int = CHECK_NEW,
                      samplings: Sequence[Sampling] = CHECK_SAMPLINGS, timeout: float = 600.0) -> CheckResult:
    """The check of serve --parallel at start: the requests one at a time (each runs alone on the carrier, the single path), then all
    of them at once (in the stream slots); every request must get the same ids both ways.  An engine failure raises."""
    cases = [([int(x) for x in p], samplings[i % len(samplings)]) for i, p in enumerate(prompts)]
    t0 = time.perf_counter()

    def one(prompt, sampling):
        with ScheduledEngine(sched) as se:
            se.set_sampling(sampling)
            return se.generate(prompt, max_new, stop_ids=None)

    alone = [one(p, s) for p, s in cases]
    together: List[Optional[GenResult]] = [None] * len(cases)
    errors: List[Optional[BaseException]] = [None] * len(cases)
    start = threading.Barrier(len(cases))

    def work(i):
        try:
            start.wait(timeout)
            together[i] = one(*cases[i])
        except BaseException as ex:                          # noqa: BLE001  (raised below, on the caller's thread)
            errors[i] = ex

    threads = [threading.Thread(target=work, args=(i,), name="yasakani-check-%d" % i, daemon=True) for i in range(len(cases))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
    for ex in errors:
        if ex is not None:
            raise ex
    if any(t.is_alive() for t in threads) or any(r is None for r in together):
        raise YasakaniError("the check of several requests at once did not finish within %.0f s" % timeout)
    res = compare_runs(alone, together)
    res.seconds = time.perf_counter() - t0
    return res
