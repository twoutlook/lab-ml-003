"""把四個 policy 的窮舉結果寫成 checkpoints/benchmark.json，給影片和 artifact 讀。

    python benchmark.py

evaluate.py 是印給人看的，這支是印給程式讀的——影片的結果頁和 artifact 的表格
都從這個 json 讀，所以影片上的數字不可能跟實際跑出來的不一樣。

因為環境是決定性的、狀態只有 512 個，這裡是「窮舉」不是「取樣」：
從每一個非終點狀態各跑一次，沒有 seed，也沒有信賴區間可談。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from env import load_config
from evaluate import build_obs, rollout
from model import NEG_INF, DuelingQNet
from solver import all_states, distance_batch, full_state, optimal_action

HERE = Path(__file__).resolve().parent
OUT = HERE / "checkpoints" / "benchmark.json"


def summarize(name, note, steps, solved, _dist, opt, live, full_row):
    """steps/solved/_dist 直接來自 rollout() 的三元組，所以呼叫端可以用 *rollout(...)。"""
    ok = solved & live
    exact = ok & (steps == opt)
    return {
        "name": name,
        "note": note,
        "solve_rate": round(float(solved[live].mean()), 4),
        "optimal_rate": round(float(exact[live].mean()), 4),
        "mean_ratio": round(float((steps[ok] / opt[ok]).mean()), 3) if ok.any() else None,
        "full_solved": bool(solved[full_row]),
        "full_steps": int(steps[full_row]) if solved[full_row] else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "checkpoints" / "best.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config()
    n = int(cfg["rings"])
    S0 = all_states(n)
    opt = distance_batch(S0)
    live = opt > 0
    full_row = int(distance_batch(full_state(n)[None])[0])
    rng = np.random.default_rng(0)
    rows = []

    def random_legal(S, mask, t):
        w = mask / np.maximum(1, mask.sum(1, keepdims=True))
        u = rng.random((S.shape[0], 1))
        return (u > w.cumsum(1)).sum(1)

    rows.append(summarize("亂走 random", "從合法動作裡均勻抽一個",
                          *rollout(random_legal, cfg, S0), opt, live, full_row))

    def greedy_off(S, mask, t):
        removable = mask * S
        return np.where(removable.any(1), removable.argmax(1), mask.argmax(1))

    rows.append(summarize("貪心拿環 greedy", "能拿下來就拿下來，逼不得已才裝回去",
                          *rollout(greedy_off, cfg, S0), opt, live, full_row))

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    dev = torch.device(args.device)
    net = DuelingQNet(ck["obs_size"], ck["n_actions"], ck["hidden"]).to(dev).eval()
    net.load_state_dict(ck["net"])

    @torch.no_grad()
    def dqn(S, mask, t):
        o = torch.as_tensor(build_obs(S, mask, t, int(cfg["maxSteps"])), device=dev)
        q = net(o).masked_fill(torch.as_tensor(mask, device=dev) < 0.5, NEG_INF)
        return q.argmax(1).cpu().numpy()

    rows.append(summarize(f"DQN agent（{ck.get('step', 0):,} steps）", "只給獎勵，沒給規則",
                          *rollout(dqn, cfg, S0), opt, live, full_row))

    def optimal(S, mask, t):
        return np.array([optimal_action(s) if s.any() else 0 for s in S])

    rows.append(summarize("最優解 optimal", "格雷碼算出來的，不是學的",
                          *rollout(optimal, cfg, S0), opt, live, full_row))

    data = {
        "rings": n,
        "starts": int(live.sum()),
        "max_steps": int(cfg["maxSteps"]),
        "full_distance": full_row,
        "trained_steps": int(ck.get("step", 0)),
        "deterministic": True,
        "rows": rows,
    }
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"寫出 {OUT}\n")
    print(f"{'policy':<28} {'解開':>7} {'剛好最優':>9} {'平均倍率':>9} {'全上':>7}")
    print("-" * 66)
    for r in rows:
        fs = f"{r['full_steps']} 步" if r["full_steps"] else "沒解開"
        print(f"{r['name']:<28} {r['solve_rate']:>7.1%} {r['optimal_rate']:>9.1%} "
              f"{(r['mean_ratio'] or float('nan')):>8.2f}x {fs:>7}")


if __name__ == "__main__":
    main()
