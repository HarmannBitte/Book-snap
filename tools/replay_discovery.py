"""Zero-execution replay over the discovery tree (Dream-RSI-style log).

docs/discovery_tree.json records every exploration decision of this project
together with the outcome it actually realized - the tree IS the simulator
(dream-rsi.com: "history is already a simulator"). Alternative ideas can be
screened against it without re-running anything:

    python tools/replay_discovery.py rank     # ideas by measured impact
    python tools/replay_discovery.py path     # the kept lineage, round by round
    python tools/replay_discovery.py dead     # what was tried and dropped, why
    python tools/replay_discovery.py open     # declared-open fronts
    python tools/replay_discovery.py query WORD
    python tools/replay_discovery.py render   # (re)write docs/DISCOVERY_LOG.md

Impact scores are CURATED (see IMPACT): measured deltas where a bench moved
(recall/precision points), capability unlocks otherwise; every score carries
its evidence string in the node's metric field.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.join(HERE, "..", "docs", "discovery_tree.json")

# curated impact: verified-recall points on the real shelf first, then
# precision guards, then capability unlocks (evidence in node.metric)
IMPACT = {
    "r16-fix1": 0.35,      # 'run' (the advertised one-shot path) worked again
    "r16-fix2": 0.20,      # RAM-safe spines reachable from run
    "r16-demo": 0.15,      # live end-to-end + offline repro both green
    "r14-scorer2": 0.11,   # real-shelf verified .22 -> .33
    "r9-budget": 0.11,     # spine-dense verified .11 -> .22
    "r8-group": 0.46,      # synth verified 0 -> .46 (multi-word candidates exist)
    "r8-synth": 0.79,      # new offline bench at .79 + CI gate
    "r14-silence": 0.90,   # 18 non-book FPs avoided (precision guard)
    "r11-weak": 0.90,      # same-title-different-book export guard
    "r5-xref-veto": 0.25,  # exported precision 3/4 -> 4/4
    "r8-spineonly": 0.40,  # shelf-only runs verify at all
    "r12-deno": 0.70,      # YouTube search unlocked
    "r13-captions": 0.30,  # free ASR, better proper nouns
    "r10-degarble": 0.05,  # +Obama via corrected garble; robustness
    "r13-llmfix": 0.30,    # gate that makes captions safe
    "r14-manual-llm": 0.11,  # the pass that moved .22 -> .33
}


def load():
    return json.load(open(TREE))


def nodes(d):
    return {n["id"]: n for n in d["nodes"]}


def cmd_rank(d):
    rows = sorted(((IMPACT.get(n["id"], 0.05 if n["verdict"] == "kept" else 0.0), n)
                   for n in d["nodes"]), key=lambda r: -r[0])
    for score, n in rows:
        if n["verdict"] != "kept" or score <= 0:
            continue
        print(f"{score:.2f}  {n['id']:<16} {n['intent'][:52]:<52} [{n['metric']}]")


def cmd_path(d):
    by = nodes(d)
    for n in d["nodes"]:
        if n["verdict"] != "kept":
            continue
        chain = []
        cur = n
        while cur:
            chain.append(cur["id"])
            cur = by.get(cur.get("parent") or "")
        print(f"r{n['round']:>2} {' <- '.join(reversed(chain))}")
        print(f"      {n['action']}  =>  {n['observation']}  [{n['metric']}]")


def cmd_filter(d, verdict):
    for n in d["nodes"]:
        if n["verdict"] == verdict:
            print(f"{n['id']:<16} {n['intent'][:48]:<48} {n['observation'][:60]}")


def cmd_query(d, word):
    w = word.lower()
    for n in d["nodes"]:
        blob = " ".join(str(n.get(k, "")) for k in
                        ("intent", "action", "observation", "metric")).lower()
        if w in blob:
            print(f"{n['id']:<16} [{n['verdict']}] {n['intent'][:60]}")


def cmd_render(d):
    by = nodes(d)
    L = ["# Discovery Log (Dream-RSI-style)",
         "",
         "Every exploration decision with its realized outcome; the tree is a",
         "replay simulator - screen ideas with `tools/replay_discovery.py`",
         "instead of re-running them. Verdicts: kept / dropped / open.",
         ""]
    for rnd in sorted({n["round"] for n in d["nodes"]}, key=lambda r: (r == "x", r == "open", r)):
        title = {"x": "Dead ends (tried, dropped, why)",
                 "open": "Declared-open fronts"}.get(rnd, f"Round {rnd}")
        L.append(f"## {title}")
        L.append("")
        for n in d["nodes"]:
            if n["round"] != rnd:
                continue
            imp = IMPACT.get(n["id"])
            L.append(f"- **{n['id']}** [{n['verdict']}"
                     + (f", impact {imp:.2f}" if imp else "") + "]"
                     + (f" commit {n['commit']}" if n.get("commit") else ""))
            L.append(f"  - intent: {n['intent']}")
            L.append(f"  - action: {n['action']}")
            L.append(f"  - outcome: {n['observation']} (metric: {n['metric']})")
            par = by.get(n.get("parent") or "")
            if par:
                L.append(f"  - branched from: {par['id']}")
        L.append("")
    L.append("## Ranked ideas (curated impact, evidence in metrics)")
    L.append("")
    for score, n in sorted(((IMPACT.get(n["id"], 0), n) for n in d["nodes"]),
                           key=lambda r: -r[0]):
        if score > 0:
            L.append(f"1. {score:.2f} `{n['id']}` - {n['intent']} [{n['metric']}]")
    out = os.path.join(HERE, "..", "docs", "DISCOVERY_LOG.md")
    open(out, "w").write("\n".join(L) + "\n" + EXTERNAL)
    print("wrote", out)


EXTERNAL = """
## External prior art consulted

- **dream-rsi.com / zhengkid/Dream-RSI** (technical report): the logging
  discipline adopted here - history as a replay simulator, decisions recorded
  with realized outcomes, alternatives screened at zero executions.
- **sonoxo/GPT-DOUG-Dream-RSI** (fork of the above): adds a Zyra/XUNIA swarm
  ecosystem stack, a DOJ NSD compliance-mapping pipeline with tests, a 24h
  compliance-audit CI workflow on integration pushes, and a compliance domain
  map. Assessment for this repo: the audit-on-push pattern is already present
  here as the offline bench gates; the swarm stack and the DOJ mapper are
  domain-specific weight with no analogue in book extraction - NOT adapted.
  Meta-lesson recorded: stacking unrelated ecosystems onto a discovery repo
  produces noise commits; this log stays single-project.
- **Panniantong/Agent-Reach** (round 12): yt-dlp JS-runtime diagnosis adopted
  (deno + --js-runtimes), which unlocked YouTube search.
"""


def main(argv):
    d = load()
    cmd = argv[1] if len(argv) > 1 else "rank"
    if cmd == "rank":
        cmd_rank(d)
    elif cmd == "path":
        cmd_path(d)
    elif cmd == "dead":
        cmd_filter(d, "dropped")
    elif cmd == "open":
        cmd_filter(d, "open")
    elif cmd == "query":
        cmd_query(d, argv[2])
    elif cmd == "render":
        cmd_render(d)
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
