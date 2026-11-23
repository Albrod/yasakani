# SPDX-License-Identifier: Apache-2.0
"""0.3.4: --prefill auto on a 24 GB-class card (yasakani/gpu.py: prefill_choice, use_prefill, plan_given_context).

A user measured the new prompt reading on an RTX 3090 Ti (24,563 MiB) with the context given by --max-ctx (issue #24): 1.39 to 1.40 times
as fast at 8,224 and 32,797 tokens with the same self-check numbers up to 131,072; at 262,144 the engine found no room for its work area.
So auto takes the new reading on a card of AUTO_NEW_MIN_TOTAL_MIB (22,000) to FULL_CONTEXT_MIN_TOTAL_MIB (30,000) when a context is
given and the free memory left beside it is at least NEW_PREFILL_RESERVE_MIB (what --prefill new plans with); in every other case auto
stays as before.  nvidia-smi is not run: its rows are given as text.
"""
import unittest

from tests.test_gpu import FASTEST, MAXQ, QUALITY, FakePack, det_of, smi_line
from yasakani import gpu as G

R3090TI = "NVIDIA GeForce RTX 3090 Ti"


def card(total, used, name=R3090TI, cc="8.6"):
    return det_of(smi_line(0, name, cc, total, used))


def given(sizes, det, ctx, mode="auto"):
    """The plan `yasakani chat --max-ctx ctx` makes (cli._open_stack: plan_given_context with the reserve of the mode, use_prefill)."""
    return G.use_prefill(G.plan_given_context(FakePack(sizes), det, ctx, G.prefill_reserve_mib(mode, det)), mode, det)


class AutoOn24GbTests(unittest.TestCase):

    def test_reporters_card(self):
        """The RTX 3090 Ti of #24 (24,563 MiB, desktop about 1.1 GB, quality model): new up to 196,608, the 0.1.7 reading at 262,144."""
        det = card(24563, 1100)
        want = {40960: "new", 65536: "new", 131072: "new", 196608: "new", 262144: "old"}
        for ctx, reading in want.items():
            plan = given(QUALITY, det, ctx)
            self.assertEqual(plan.prefill, reading, ctx)
            self.assertEqual(plan.max_ctx, ctx)                                  # the context is never changed
            self.assertEqual(plan.room_mib >= G.NEW_PREFILL_RESERVE_MIB, reading == "new", (ctx, plan.room_mib))
            if reading == "new":
                self.assertEqual({k: plan.env[k] for k in ("YSKN_Q35_PMAX", "YSKN_Q35_LAYER_ROWS")}, G.SMALL_CARD_ENV)
                self.assertNotIn("YSKN_Q35_PF_V2", plan.env)
                self.assertEqual(plan.reserve_mib, G.NEW_PREFILL_RESERVE_MIB)
            else:
                self.assertEqual(plan.env["YSKN_Q35_PF_V2"], "0")
                self.assertEqual(plan.reserve_mib, 0)
        self.assertEqual(given(QUALITY, det, 131072).room_mib, 3601)
        self.assertEqual(given(QUALITY, det, 262144).room_mib, 801)

    def test_room_is_the_reserve_of_new(self):
        """The room is what fit_tokens leaves: the same context fits with the reserve of --prefill new exactly when auto takes new."""
        det = card(24564, 1500)
        for sizes in (FASTEST, QUALITY, MAXQ):
            for ctx in range(16384, 262144 + 1, 16384):
                plan = given(sizes, det, ctx)
                if plan.env.get("YSKN_BDR_SHEAD") == "0":                       # the head does not fit even without a reserve
                    self.assertEqual(plan.prefill, "old", ctx)
                    continue
                fits = G.fit_tokens(det.gpu.free_mib - G.NEW_PREFILL_RESERVE_MIB, sum(sizes), G.overhead_mib(sizes[0])) >= ctx
                self.assertEqual(plan.prefill == "new", fits, (sizes[0], ctx, plan.room_mib))

    def test_the_limits(self):
        """22,000 MiB counts, 21,999 does not; 1,800 MiB of room counts, 1,799 does not."""
        self.assertEqual(G.prefill_choice("auto", card(22000, 0), 5000, 65536)[0], "new")
        self.assertEqual(G.prefill_choice("auto", card(21999, 0), 5000, 65536)[0], "old")
        self.assertEqual(G.prefill_choice("auto", card(24564, 0), G.NEW_PREFILL_RESERVE_MIB, 65536)[0], "new")
        self.assertEqual(G.prefill_choice("auto", card(24564, 0), G.NEW_PREFILL_RESERVE_MIB - 1, 65536)[0], "old")
        self.assertEqual(G.prefill_choice("auto", card(24564, 0), 5000, None)[0], "old")       # no --max-ctx
        self.assertEqual(G.prefill_choice("auto", card(24564, 0), None, 65536)[0], "old")      # room not known

    def test_the_reason_in_the_line(self):
        """The start-up line says "no --max-ctx" only when none was given; with a context but no known room it names the card only."""
        try_new = "--prefill new or YASAKANI_PREFILL=new to try it"
        self.assertEqual(G.prefill_choice("auto", card(24564, 0), 5000, None)[2],
                         "prompt reading: as in 0.1.7 (setting auto: 24564 MiB card and no --max-ctx; %s)" % try_new)
        self.assertEqual(G.prefill_choice("auto", card(24564, 0), None, 65536)[2],
                         "prompt reading: as in 0.1.7 (setting auto: 24564 MiB card; %s)" % try_new)

    def test_without_max_ctx_as_before(self):
        """No --max-ctx: the context the package chooses, and the reading of 0.1.7 (auto gives up no context)."""
        det = card(24563, 1100)
        for sizes in (FASTEST, QUALITY, MAXQ):
            plain = G.plan_context(FakePack(sizes), det)
            plan = G.use_prefill(G.plan_context(FakePack(sizes), det, G.prefill_reserve_mib("auto", det)), "auto", det)
            self.assertEqual((plan.max_ctx, plan.prefill, plan.reserve_mib), (plain.max_ctx, "old", 0))
            self.assertEqual(plan.env["YSKN_Q35_PF_V2"], "0")
        self.assertEqual(G.prefill_reserve_mib("auto", det), 0)
        plan = G.use_prefill(G.plan_context(FakePack(QUALITY), det), "auto", det)
        self.assertEqual(plan.prefill_line, "prompt reading: as in 0.1.7 (setting auto: 24563 MiB card and no --max-ctx; "
                                            "--prefill new or YASAKANI_PREFILL=new to try it)")

    def test_other_cards_untouched(self):
        """Under 22,000 MiB, from 30,000 MiB, of unknown memory, and the settings new and old: as in 0.3.3."""
        cases = [(card(16311, 900, "NVIDIA GeForce RTX 5060 Ti", "12.0"), FASTEST, 16384),
                 (card(16303, 500, "NVIDIA GeForce RTX 5080", "12.0"), FASTEST, 32768),
                 (card(32607, 1107, "NVIDIA GeForce RTX 5090", "12.0"), QUALITY, 131072),
                 (det_of("0, GPU-1, NVIDIA GeForce RTX 4080, 8.9, [N/A], [N/A], 581.57"), FASTEST, 32768)]
        for det, sizes, ctx in cases:
            plan = given(sizes, det, ctx)
            self.assertEqual(plan.prefill_line, G.prefill_choice("auto", det)[2], plan.prefill_line)
            self.assertEqual(plan.prefill, G.prefill_choice("auto", det)[0])
        r = card(24563, 1100)
        for mode in ("new", "old"):
            for ctx in (65536, 262144):
                plan = given(QUALITY, r, ctx, mode)
                self.assertEqual((plan.prefill, plan.prefill_line), G.prefill_choice(mode, r)[0::2], (mode, ctx))
        self.assertEqual(given(QUALITY, card(16311, 900, "NVIDIA GeForce RTX 5060 Ti", "12.0"), 16384).prefill_line,
                         "prompt reading: as in 0.1.7 (setting auto: 16311 MiB card, new only from 30000 MiB; "
                         "--prefill new or YASAKANI_PREFILL=new to try it)")

    def test_lines(self):
        """The start-up line says which reading, why, and how to take the other one."""
        det = card(24563, 1100)
        self.assertEqual(given(QUALITY, det, 131072).prefill_line,
                         "prompt reading: new (setting auto: 24563 MiB card, 3601 MiB free beside --max-ctx 131072; "
                         "--prefill old for the 0.1.7 one)")
        self.assertEqual(given(QUALITY, det, 262144).prefill_line,
                         "prompt reading: as in 0.1.7 (setting auto: 24563 MiB card, 801 MiB free beside --max-ctx 262144, "
                         "new needs 1800; --prefill new to try it)")
        for ctx in (16384, 131072, 262144):
            line = given(QUALITY, det, ctx).prefill_line
            self.assertLess(len(line), 140, line)
            self.assertNotIn("\n", line)

    def test_head_left_out_counts(self):
        """When the draft model's head does not fit beside the given context it is left out, and the room is counted without it."""
        det = card(24564, 3500)
        plan = given(MAXQ, det, 163840)
        self.assertEqual(plan.env.get("YSKN_BDR_SHEAD"), "0")
        self.assertEqual(plan.prefill, "old")
        self.assertLess(plan.room_mib, G.NEW_PREFILL_RESERVE_MIB)

    def test_inkling2_and_parallel_unchanged(self):
        """An Inkling-2 folder and serve --parallel plans have no room_mib: auto stays with the reading of 0.1.7 there."""
        det = card(24563, 1100)
        pack = FakePack(QUALITY)
        pack.is_hf_inkling2 = True
        self.assertIsNone(G.plan_given_context(pack, det, 32768).room_mib)
        par = G.use_prefill(G.plan_given_parallel(FakePack(QUALITY), det, 32768, 2), "auto", det)
        self.assertEqual(par.prefill, "old")


if __name__ == "__main__":
    unittest.main()
