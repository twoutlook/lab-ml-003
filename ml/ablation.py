"""跑（或收集）reward shaping 的三組對照實驗，寫成 checkpoints/ablation.json。

    python ablation.py              # 缺哪組就跑哪組，然後收集
    python ablation.py --collect    # 只收集，不重跑
    python ablation.py --force      # 三組全部重跑

三組差別只有 shaping 的設定，其他超參數完全一樣：

    ab_fixed     shapingGamma = 1.0     本專案的做法
    ab_textbook  shapingGamma = gamma   教科書寫法（Ng et al. 1999 的保證條件）
    ab_sparse    shaping = 0            完全不 shaping，只有終點那 +1

影片和 artifact 上的那三條曲線就是這個檔案的輸出。它們是真的跑出來的，不是畫的。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CKPT = HERE / "checkpoints"
OUT = CKPT / "ablation.json"

STEPS = 300_000

RUNS = [
    {"tag": "ab_fixed", "name": "shapingGamma = 1", "setting": "本專案的做法",
     "args": ["--shaping-gamma", "1.0"]},
    {"tag": "ab_textbook", "name": "shapingGamma = γ", "setting": "教科書寫法（Ng et al. 1999）",
     "args": ["--shaping-gamma", "0.95"]},
    {"tag": "ab_sparse", "name": "shaping = 0", "setting": "純稀疏獎勵，γ = 0.999",
     "args": ["--shaping", "0", "--gamma", "0.999"]},
]


def run_one(r):
    print(f"\n=== {r['tag']}  {r['name']} ({r['setting']}) ===", flush=True)
    subprocess.run(
        [sys.executable, str(HERE / "train.py"), "--steps", str(STEPS),
         "--eval-every", "25000", "--tag", r["tag"], *r["args"]],
        cwd=str(HERE), check=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="只收集現有結果")
    ap.add_argument("--force", action="store_true", help="三組全部重跑")
    args = ap.parse_args()

    for r in RUNS:
        hist = CKPT / r["tag"] / "history.json"
        if args.force and (CKPT / r["tag"]).exists():
            shutil.rmtree(CKPT / r["tag"])
        if not hist.exists() and not args.collect:
            run_one(r)

    runs = []
    for r in RUNS:
        hist = CKPT / r["tag"] / "history.json"
        if not hist.exists():
            sys.exit(f"缺 {hist} —— 先跑 python ablation.py（不加 --collect）")
        h = json.loads(hist.read_text(encoding="utf-8"))
        runs.append({
            "tag": r["tag"], "name": r["name"], "setting": r["setting"],
            "steps": STEPS,
            "final_optimal_rate": h[-1]["optimal_rate"],
            "final_solve_rate": h[-1]["solve_rate"],
            "best_optimal_rate": max(x["optimal_rate"] for x in h),
            "history": [{"step": x["step"], "optimal_rate": x["optimal_rate"],
                         "solve_rate": x["solve_rate"]} for x in h],
        })

    OUT.write_text(json.dumps({"steps": STEPS, "runs": runs}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n寫出 {OUT}\n")
    print(f"{'設定':<24} {'最後 optimal':>13} {'最後 solve':>12} {'歷程最佳':>10}")
    print("-" * 64)
    for r in runs:
        print(f"{r['name'] + ' · ' + r['setting']:<24} {r['final_optimal_rate']:>12.1%} "
              f"{r['final_solve_rate']:>11.1%} {r['best_optimal_rate']:>9.1%}")


if __name__ == "__main__":
    main()
