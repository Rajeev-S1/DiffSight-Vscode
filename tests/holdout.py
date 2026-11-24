"""Held-out randomized evaluation. New text, random edits / objects / scan degradation per seed;
nothing here was used when tuning thresholds.   python tests/holdout.py [n_cases] [seed0]"""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_suite as M  # noqa: E402
import benchmark as B  # noqa: E402
import samples as S  # noqa: E402

WORDS = ("harbor lantern quarterly ledger vessel compliance schedule renewal amendment tenant landlord premises "
         "utility deposit warranty shipment carrier customs tariff inspection clause appendix witness notary "
         "arbitration venue liability indemnity coverage premium claimant policy broker pledge collateral lender "
         "borrower interest tenure covenant forfeiture delivery vendor invoice remittance penalty cargo freight").split()
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def sentence(rng, with_number=False):
    n = rng.randint(8, 13)
    ws = [rng.choice(WORDS) for _ in range(n)]
    ws[0] = ws[0].capitalize()
    if with_number:
        ws.insert(rng.randint(2, n - 2), str(rng.randint(1000, 99999)))
    return " ".join(ws) + "."


def build_case(rng):
    lines = [sentence(rng, with_number=(i % 3 == 0)) for i in range(9)]
    a = M.base_page(lines=lines, title=" ".join(w.capitalize() for w in rng.sample(WORDS, 3)),
                    header=f"{rng.choice(WORDS).capitalize()} Holdings - Draft", footer="Page 1 of 1", table=rng.random() < 0.5)
    b = copy.deepcopy(a)
    truth, used = [], set()
    kinds = rng.sample(["word", "char", "punct", "num", "stamp_add", "stamp_rm", "sig_add", "sig_change"], k=rng.randint(3, 5))
    for k in kinds:
        li = rng.choice([i for i in range(9) if i not in used])
        lid = f"L{li}"
        ia, ib = M.find(a, lid), M.find(b, lid)
        s = ia["s"]
        if k == "word":
            cands = [w for w in s.rstrip(".").split() if len(w) >= 5 and w.isalpha()]
            old = rng.choice(cands)
            new = rng.choice([w for w in WORDS if w != old])
            truth.append(M.T(bbox_a=M.sub_box(ia, old), kinds=M.TEXT_K, desc=f"{old}->{new}"))
            ib["s"] = s.replace(old, new, 1)
        elif k == "char":
            cands = [w for w in s.rstrip(".").split() if len(w) >= 6 and w.isalpha()]
            old = rng.choice(cands)
            p = rng.randint(1, len(old) - 2)
            new = old[:p] + rng.choice([c for c in LETTERS if c != old[p]]) + old[p + 1:]
            truth.append(M.T(bbox_a=M.sub_box(ia, old), kinds=M.TEXT_K, desc=f"{old}->{new}"))
            ib["s"] = s.replace(old, new, 1)
        elif k == "punct":
            last = s.split()[-1]
            new = last[:-1] + rng.choice([",", ";", ":", "!", "?", ""])
            truth.append(M.T(bbox_a=M.sub_box(ia, last), kinds=["Punctuation", "Text"], desc=f"{last}->{new}"))
            ib["s"] = s[: -len(last)] + new
        elif k == "num":
            nums = [w for w in s.split() if w.isdigit()]
            if not nums:
                continue
            old = nums[0]
            new = str(int(old) + rng.choice([1, 10, 100, 900]))
            truth.append(M.T(bbox_a=M.sub_box(ia, old), kinds=M.TEXT_K, desc=f"{old}->{new}"))
            ib["s"] = s.replace(old, new, 1)
        used.add(li)
    # objects in the free lower-right area (never over text), distinct slots
    slots = [(330, 560), (430, 640), (120, 690)]
    rng.shuffle(slots)
    for k in kinds:
        if not k.startswith(("stamp", "sig")):
            continue
        if not slots:
            continue
        x, y = slots.pop()
        if k == "stamp_add":
            st = M.stamp_item(rng.choice(["blue", "red", "green"]), x, y)
            b["items"].append(st)
            truth.append(M.T(bbox_a=M.rect_px(x, y, 72, 72), bbox_b=M.rect_px(x, y, 72, 72), kinds=["Stamp"], category="Added", desc="stamp added"))
        elif k == "stamp_rm":
            a["items"].append(M.stamp_item(rng.choice(["blue", "red", "green"]), x, y))
            truth.append(M.T(bbox_a=M.rect_px(x, y, 72, 72), kinds=["Stamp"], category="Removed", desc="stamp removed"))
        elif k == "sig_add":
            b["items"].append(M.sig_item(rng.choice([1, 2]), x, y))
            truth.append(M.T(bbox_a=M.rect_px(x, y, 125, 41), bbox_b=M.rect_px(x, y, 125, 41), kinds=["Signature", "Handwriting"], category="Added", desc="signature added"))
        elif k == "sig_change":
            a["items"].append(M.sig_item(1, x, y))
            b["items"].append(M.sig_item(2, x, y))
            truth.append(M.T(bbox_a=M.rect_px(x, y, 125, 41), kinds=["Signature", "Handwriting"], desc="signature changed"))
    return a, b, truth


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    out = Path(__file__).resolve().parent.parent / "data" / "holdout"
    out.mkdir(parents=True, exist_ok=True)
    tot = dict(truth=0, tp=0, tpu=0, fp=0, conf=0, unc=0)
    rows = []
    print(f"{'case':14s} {'mode':8s} {'truth':>5s} {'TP':>3s} {'+unc':>4s} {'FP':>3s} {'conf':>4s} {'sec':>5s}  notes")
    for i in range(n):
        seed = seed0 + i
        rng = random.Random(seed)
        a, b, truth = build_case(rng)
        d = out / f"h{seed}"
        d.mkdir(exist_ok=True)
        M.render_pdf(d / "src_A.pdf", [a])
        M.render_pdf(d / "src_B.pdf", [b])
        mode = rng.choice(["digital", "scan", "scan", "mixed"])
        deg = lambda s: dict(angle=rng.uniform(-1.2, 1.2), noise=rng.uniform(3, 8), blur=rng.uniform(0.6, 1.2), jpeg=rng.randint(45, 80),
                             shift=(rng.randint(-12, 12), rng.randint(-12, 12)), scale=rng.uniform(0.99, 1.01),
                             shadow=rng.choice([0, 0.1, 0.2]), seed=s)
        for side, srcname in (("A", "src_A.pdf"), ("B", "src_B.pdf")):
            if mode == "digital" or (mode == "mixed" and side == "A"):
                (d / f"{side}.pdf").write_bytes((d / srcname).read_bytes())
            else:
                M.to_scan(d / srcname, d / f"{side}.pdf", "pdf", **deg(seed * 3 + (side == "B")))
        r = B.run_one(d, d / "out")
        if "error" in r:
            print(f"h{seed}: ERROR {r['error'][-200:]}")
            continue
        s = B.score(r["report"], truth)
        rows.append({"seed": seed, "mode": mode, **s, "seconds": r["seconds"]})
        for k, v in (("truth", "truth"), ("tp", "tp"), ("tpu", "tp_incl_uncertain"), ("fp", "fp"), ("conf", "confirmed"), ("unc", "uncertain")):
            tot[k] += s[v]
        note = ("MISSED: " + "; ".join(s["missed"]) if s["missed"] else "") + (" FP: " + " ## ".join(s["fp_list"][:2]) if s["fp_list"] else "")
        print(f"h{seed:<13d} {mode:8s} {s['truth']:5d} {s['tp']:3d} {s['tp_incl_uncertain']:4d} {s['fp']:3d} {s['confirmed']:4d} {r['seconds']:5.1f}  {note}")
    summary = {"cases": len(rows), "truth": tot["truth"], "recall_confirmed": round(tot["tp"] / max(1, tot["truth"]), 3),
               "recall_incl_uncertain": round(tot["tpu"] / max(1, tot["truth"]), 3), "false_positives": tot["fp"],
               "precision": round((tot["conf"] - tot["fp"]) / max(1, tot["conf"]), 3), "uncertain_findings": tot["unc"]}
    print("\nHOLDOUT SUMMARY", json.dumps(summary))
    (out / "results.json").write_text(json.dumps({"summary": summary, "cases": rows}, indent=1))


if __name__ == "__main__":
    main()
