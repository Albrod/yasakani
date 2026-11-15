#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""tables_0114.py - the thinking-off speed table of the README with YASAKANI measured again with 0.1.4 (November 15, 2026), and
0.1.2 / 0.1.3 / 0.1.4 side by side.

  python tools/speed/tables_0114.py [--out DIR]          (Python 3.8+, standard library only)

Reads   bench/20261115/results/<version>/n12_ysk_<model>_v8/{n12,own,own_plus}   YASAKANI 0.1.2, 0.1.3 and 0.1.4, both models,
                                                                                 measured one after the other with the clients of
                                                                                 bench/20261108 (thinking off, greedy, 2 runs)
        bench/20261108/results/...                                               the eight other engines (November 3 and 6), as before
Everything else is as in tools/speed/tables_strict.py: the same speed of a run, the same averages, the same stricter rule for
repetition (it imports that program and bench/20261108/scripts/tables.py).  "own_plus" holds one extra task that is not a column of
the README table: the long JSONL edit of bench/20261115/prompts/cases_plus.json (160 records returned with 3 changed).

Writes SPEED_TABLES_0114.md and speed_tables_0114.json to --out (default: the folder of this file).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import tables_strict as S  # noqa: E402

T = S.T
NEW = HERE.parents[1] / "bench" / "20261115" / "results"
VERSIONS = ["v0.1.2", "v0.1.3", "v0.1.4"]
MODELS = [("YASAKANI fastest model", "speed"), ("YASAKANI quality model", "quality")]


def ours(version, model, flag):
    """-> stats (7 columns, as tables.py) and the extra task, for one version and one model"""
    base = NEW / version / ("n12_ysk_%s_v8" % model)
    T.flag = flag
    n12 = T.n12_rows(base / "n12")
    own = T.own_rows(T.load(base / "own"))
    st = T.stats(n12, own)
    plus = T.own_rows(T.load(base / "own_plus"))
    st["long_jsonl_edit"] = statistics.mean(r["decode_tok_s"] for r in plus) if plus else None
    st["exact"] = {"%s r%d" % (r["case"], r["k"]): bool(r.get("exact")) for r in own + plus if "exact" in r}
    st["prompts"] = {fx: dict(mean=v, flags=fl) for fx, (v, fl) in S.per_prompt(n12).items()}
    T.flag = S.PUBLISHED_FLAG
    return st


def build():
    before, after, _ = S.build()
    res = {"columns": T.COLS, "engines_strict": {n: e["off"]["mean_clean"] for n, e in after["engines"].items()},
           "engines_all_runs": {n: e["off"]["mean"] for n, e in before["engines"].items()}, "yasakani": {}}
    for name, m in MODELS:
        res["yasakani"][name] = {v: ours(v, m, S.strict_flag) for v in VERSIONS}
    return res


def markdown(res):
    f = lambda v: "-" if v is None else "{:,.0f}".format(v)
    cols = T.COLS
    md = ["# Thinking off: YASAKANI measured again with 0.1.4 (November 15, 2026)", "",
          "Output of `tools/speed/tables_0114.py`. YASAKANI's rows come from `bench/20261115` (0.1.4, measured on November 15 with the",
          "clients of `bench/20261108`: thinking off, greedy, 2 runs per task); the other engines are the records of `bench/20261108`",
          "(November 3 and 6). The rule for runs that are left out is the one of `tools/speed/tables_strict.py`, the same for every engine.",
          "tok/s; conditions and limits as in the README of `bench/`.", "",
          "## The table of the README (average without repetition loops and unrequested tool calls)", "",
          "| Engine | " + " | ".join(cols) + " |", "|---|" + "---|" * 7]
    for name, _ in MODELS:
        md.append("| %s (0.1.4) | %s |" % (name, " | ".join(f(v) for v in res["yasakani"][name]["v0.1.4"]["mean_clean"])))
    md += ["| %s | %s |" % (n, " | ".join(f(v) for v in vals)) for n, vals in res["engines_strict"].items()]
    best = T.colmax(list(res["engines_strict"].values()))
    md.append("| (largest value of the other engines) | %s |" % " | ".join(f(v) for v in best))
    for name, _ in MODELS:
        md.append("| %s (0.1.4) / largest value of the others | %s |" % (name, " | ".join(
            "%.2f" % (a / b) for a, b in zip(res["yasakani"][name]["v0.1.4"]["mean_clean"], best))))
    md += ["", "## Average of all runs (no run left out)", "", "| Engine | " + " | ".join(cols) + " |", "|---|" + "---|" * 7]
    for name, _ in MODELS:
        md.append("| %s (0.1.4) | %s |" % (name, " | ".join(f(v) for v in res["yasakani"][name]["v0.1.4"]["mean"])))
    md += ["| %s | %s |" % (n, " | ".join(f(v) for v in vals)) for n, vals in res["engines_all_runs"].items()]
    md += ["", "## 0.1.2, 0.1.3 and 0.1.4 one after the other", "",
           "The same clients, the same PC and model files, one server at a time (fastest model 0.1.2 -> 0.1.3 -> 0.1.4, then quality model",
           "0.1.4 -> 0.1.3 -> 0.1.2). The last column is the extra task of `bench/20261115/prompts/cases_plus.json`.", ""]
    for name, _ in MODELS:
        y = res["yasakani"][name]
        md += ["### " + name, "", "| Version | " + " | ".join(cols) + " | Long JSONL edit |", "|---|" + "---|" * 8]
        for v in VERSIONS:
            md.append("| %s | %s | %s |" % (v[1:], " | ".join(f(x) for x in y[v]["mean_clean"]), f(y[v]["long_jsonl_edit"])))
        for a, b in (("v0.1.4", "v0.1.2"), ("v0.1.3", "v0.1.2")):
            xs = ["%.2f" % (p / q) if (p and q) else "-" for p, q in zip(y[a]["mean_clean"] + [y[a]["long_jsonl_edit"]],
                                                                     y[b]["mean_clean"] + [y[b]["long_jsonl_edit"]])]
            md.append("| %s / %s | %s |" % (a[1:], b[1:], " | ".join(xs)))
        md += ["", "Runs left out: " + "; ".join("%s: %s" % (v[1:], ", ".join("%s %s" % kv for kv in sorted(y[v]["left_out"].items())) or "none")
                                                for v in VERSIONS),
               "", "Edit answers equal to the expected file (`exact`): " + "; ".join(
                   "%s: %s" % (v[1:], ", ".join("%s %s" % (k, "yes" if e else "no") for k, e in sorted(y[v]["exact"].items()))) for v in VERSIONS),
               "", "| Prompt (12 published) | " + " | ".join(v[1:] for v in VERSIONS) + " | 0.1.4 / 0.1.2 |", "|---|---|---|---|---|"]
        for fx in sorted(y["v0.1.2"]["prompts"]):
            cells = []
            for v in VERSIONS:
                p = y[v]["prompts"][fx]
                cells.append(f(p["mean"]) if p["mean"] is not None else "left out (%s)" % p["flags"])
            a, b = y["v0.1.4"]["prompts"][fx]["mean"], y["v0.1.2"]["prompts"][fx]["mean"]
            md.append("| %s | %s | %s |" % (fx.replace("scenario_", ""), " | ".join(cells), "%.3f" % (a / b) if (a and b) else "-"))
        md.append("")
    md += ["The thinking-on tables were not measured again; they are those of `bench/20261108/tables/SPEED_TABLES.md` (the engine of",
           "0.1.0 to 0.1.2, the build of November 7).", ""]
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()
    res = build()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md = markdown(res)
    for name, text in (("speed_tables_0114.json", json.dumps(res, ensure_ascii=False, indent=1) + "\n"), ("SPEED_TABLES_0114.md", md)):
        with open(out / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    sys.stdout.write(md)


if __name__ == "__main__":
    main()
