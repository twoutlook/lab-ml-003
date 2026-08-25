"""評估一份 checkpoint，並跟三個基準線比較。

    python evaluate.py                  # best.pt vs 亂走 vs 貪心 vs 最優解
    python evaluate.py --ckpt checkpoints/last.pt

這個專案的評估有一個打磚塊沒有的奢侈品：狀態只有 2^9 = 512 個，
而且每個狀態的最優步數都算得出來。所以我們可以「窮舉」——
從全部 512 個起點各跑一次，看策略在哪些地方會走錯。沒有取樣雜訊。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from env import load_config
from model import DuelingQNet, NEG_INF
from solver import all_states, distance_batch, full_state, legal_mask_batch, optimal_action

HERE = Path(__file__).resolve().parent


def build_obs(S: np.ndarray, mask: np.ndarray, steps: int, max_steps: int) -> np.ndarray:
    """跟 RingsEnv.obs() 同一套排版，只是一次做一整批。"""
    B, n = S.shape
    o = np.empty((B, 2 * n + 1), dtype=np.float32)
    o[:, :n] = S * 2.0 - 1.0
    o[:, n : 2 * n] = mask
    o[:, 2 * n] = steps / max_steps
    return o


def rollout(action_fn, cfg, starts: np.ndarray):
    """讓一個策略從每個起點各走一次，回報步數與是否解開。

    action_fn(S, mask, steps) -> 每一列的動作。整批一起走，所以 GPU 只送一次。
    """
    S = starts.copy().astype(np.int8)
    B = S.shape[0]
    max_steps = int(cfg["maxSteps"])
    dist = distance_batch(S)
    steps = np.zeros(B, dtype=np.int64)
    done = dist == 0
    for t in range(max_steps):
        if done.all():
            break
        mask = legal_mask_batch(S)
        a = action_fn(S, mask, t)
        live = ~done
        # 不合法的動作直接視為原地不動（只有亂走的基準線才可能踩到）
        ok = mask[np.arange(B), a] == 1
        move = live & ok
        S[move, a[move]] ^= 1
        steps[live] += 1
        dist = distance_batch(S)
        done = done | (dist == 0)
    return steps, done, dist


def eval_all_starts(net, cfg, device, batch_states=None):
    """從全部 512 個狀態各跑一次貪婪策略。回傳的 dict 就是訓練時記錄的指標。"""
    n = int(cfg["rings"])
    S0 = all_states(n) if batch_states is None else batch_states
    opt = distance_batch(S0)
    max_steps = int(cfg["maxSteps"])

    @torch.no_grad()
    def act(S, mask, t):
        o = torch.as_tensor(build_obs(S, mask, t, max_steps), device=device)
        q = net(o).masked_fill(torch.as_tensor(mask, device=device) < 0.5, NEG_INF)
        return q.argmax(dim=1).cpu().numpy()

    was_training = net.training
    net.eval()
    steps, solved, _ = rollout(act, cfg, S0)
    if was_training:
        net.train()

    full_row = int(distance_batch(full_state(n)[None])[0])   # 全上的狀態 = 第 341 列
    exact = solved & (steps == opt)
    live = opt > 0
    return {
        "solve_rate": float(solved[live].mean()),
        "optimal_rate": float(exact[live].mean()),
        "full_steps": int(steps[full_row]) if solved[full_row] else -1,
        "full_solved": bool(solved[full_row]),
        "mean_ratio": float((steps[solved & live] / opt[solved & live]).mean()) if (solved & live).any() else float("nan"),
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
    rows = []

    # 1) 亂走：合法動作裡均勻挑一個
    rng = np.random.default_rng(0)

    def random_legal(S, mask, t):
        w = mask / np.maximum(1, mask.sum(1, keepdims=True))
        c = w.cumsum(1)
        u = rng.random((S.shape[0], 1))
        return (u > c).sum(1)

    rows.append(("亂走（合法動作均勻）", *rollout(random_legal, cfg, S0)))

    # 2) 貪心：能拿下來就拿下來，逼不得已才裝回去
    def greedy_off(S, mask, t):
        removable = mask * S            # 這個動作是「拿下」嗎
        pick = np.where(removable.any(1), removable.argmax(1), mask.argmax(1))
        return pick

    rows.append(("貪心拿環（只想往前）", *rollout(greedy_off, cfg, S0)))

    # 3) 最優解
    def optimal(S, mask, t):
        return np.array([optimal_action(s) if s.any() else 0 for s in S])

    rows.append(("最優解（格雷碼）", *rollout(optimal, cfg, S0)))

    # 4) DQN
    p = Path(args.ckpt)
    ck = None
    if p.exists():
        dev = torch.device(args.device)
        ck = torch.load(p, map_location=dev, weights_only=False)
        net = DuelingQNet(ck["obs_size"], ck["n_actions"], ck["hidden"]).to(dev).eval()
        net.load_state_dict(ck["net"])

        @torch.no_grad()
        def dqn(S, mask, t):
            o = torch.as_tensor(build_obs(S, mask, t, int(cfg["maxSteps"])), device=dev)
            q = net(o).masked_fill(torch.as_tensor(mask, device=dev) < 0.5, NEG_INF)
            return q.argmax(1).cpu().numpy()

        rows.append((f"DQN（訓練 {ck.get('step', 0):,} steps）", *rollout(dqn, cfg, S0)))
    else:
        print(f"（找不到 {p}，先跑 train.py）")

    print(f"\n{n} 個環，最優 {opt[full_row]} 步，上限 {cfg['maxSteps']} 步")
    print(f"從全部 {live.sum()} 個非終點狀態各跑一次（沒有取樣雜訊，這個環境是決定性的）\n")
    print(f"{'policy':<26} {'全上解開':>8} {'用幾步':>7} {'解開比例':>9} {'剛好最優':>9} {'平均倍率':>9}")
    print("-" * 78)
    for name, steps, solved, _ in rows:
        exact = solved & (steps == opt)
        sr = solved[live].mean()
        orate = exact[live].mean()
        ratio = (steps[solved & live] / opt[solved & live]).mean() if (solved & live).any() else float("nan")
        fs = f"{steps[full_row]}" if solved[full_row] else "—"
        print(f"{name:<26} {'是' if solved[full_row] else '否':>8} {fs:>7} "
              f"{sr:>8.1%} {orate:>9.1%} {ratio:>9.2f}x")
    print()
    if ck is not None and ck.get("eval"):
        e = ck["eval"]
        print(f"（best.pt 存檔當下的指標：optimal_rate={e['optimal_rate']:.1%} full_steps={e['full_steps']}）\n")


if __name__ == "__main__":
    main()
