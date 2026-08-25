"""表格式 Q-learning：把 512 個狀態的 Q 值直接開成一張表。

    python tabular.py
    python tabular.py --episodes 20000 --show 12    # 跑久一點、印多一點

為什麼要有這個檔案？
    因為九連環的狀態只有 2^9 = 512 個，開表完全開得起來。
    這張表會在幾秒內收斂到 100% 最優——比 DQN 快、比 DQN 準、而且可以整張印出來看。

    所以這裡的結論是「這個問題根本不需要神經網路」。
    神經網路的價值在於狀態多到開不了表的時候（打磚塊的球座標是連續的，
    俄羅斯方塊的盤面有 2^200 種）。先看懂表格版在做什麼，
    再回頭看 DQN 就會發現：它只是在「猜」這張表而已。
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from env import load_config
from solver import all_states, distance_batch, legal_mask_batch, state_from_distance

NEG_INF = -1e9


def state_id(s) -> int:
    """狀態 -> 0..511 的編號。直接把 9 個 bit 讀成整數。"""
    v = 0
    for i, b in enumerate(s):
        v |= int(b) << i
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5_000)
    ap.add_argument("--alpha", type=float, default=0.3, help="學習率。表格版可以開很大")
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--show", type=int, default=8, help="最後印出前幾個狀態的 Q 值")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    n = int(cfg["rings"])
    gamma = float(cfg["gamma"])
    shaping_gamma = float(cfg.get("shapingGamma", 1.0))
    R = cfg["reward"]
    rng = np.random.default_rng(args.seed)

    # 先把 512 個狀態的規則全部算好，之後查表就好
    S = all_states(n)                      # 第 d 列 = 離終點 d 步的狀態
    dist = distance_batch(S)
    mask = legal_mask_batch(S)
    ids = np.array([state_id(s) for s in S])
    by_id = np.zeros((2 ** n, n), dtype=np.int8)
    id_dist = np.zeros(2 ** n, dtype=np.int64)
    id_mask = np.zeros((2 ** n, n), dtype=np.int8)
    for row in range(2 ** n):
        by_id[ids[row]] = S[row]
        id_dist[ids[row]] = dist[row]
        id_mask[ids[row]] = mask[row]

    Q = np.zeros((2 ** n, n), dtype=np.float64)
    Q[id_mask == 0] = NEG_INF              # 不合法的動作永遠是 -inf，不學也不選

    max_steps = int(cfg["maxSteps"])
    t0 = time.time()
    for ep in range(args.episodes):
        d0 = int(rng.integers(1, 2 ** n))
        sid = state_id(state_from_distance(d0, n))
        for _ in range(max_steps):
            legal = np.flatnonzero(id_mask[sid])
            if rng.random() < args.eps:
                a = int(rng.choice(legal))
            else:
                a = int(legal[np.argmax(Q[sid, legal])])
            # 走一步：翻一個 bit
            sid2 = sid ^ (1 << a)
            d, d2 = id_dist[sid], id_dist[sid2]
            r = R["step"] + R["shaping"] * (shaping_gamma * (-d2) - (-d))
            r += R["naiveRingOff"] * (1.0 if by_id[sid][a] == 1 else -1.0)
            done = d2 == 0
            if done:
                r += R["solve"]
            legal2 = np.flatnonzero(id_mask[sid2])
            best2 = 0.0 if done else float(Q[sid2, legal2].max())
            Q[sid, a] += args.alpha * (r + gamma * best2 - Q[sid, a])
            sid = sid2
            if done:
                break

    # 評估：從每個非終點狀態走貪婪策略
    solved = optimal = 0
    for d0 in range(1, 2 ** n):
        sid = state_id(state_from_distance(d0, n))
        steps = 0
        while id_dist[sid] != 0 and steps < max_steps:
            legal = np.flatnonzero(id_mask[sid])
            sid ^= 1 << int(legal[np.argmax(Q[sid, legal])])
            steps += 1
        if id_dist[sid] == 0:
            solved += 1
            if steps == d0:
                optimal += 1
    live = 2 ** n - 1
    print(f"表格 Q-learning：{args.episodes:,} 局，{time.time() - t0:.1f} 秒")
    print(f"  512 個起點：解開 {solved / live:.1%}，剛好最優 {optimal / live:.1%}")

    print(f"\n前 {args.show} 個狀態的 Q 值（依離終點的距離排）：")
    print(f"{'距離':>4}  {'狀態':>11}  {'合法動作的 Q':<34} 選誰  最優是誰")
    for d0 in range(1, args.show + 1):
        s = state_from_distance(d0, n)
        sid = state_id(s)
        legal = np.flatnonzero(id_mask[sid])
        txt = "  ".join(f"環{a + 1}={Q[sid, a]:+.3f}" for a in legal)
        pick = int(legal[np.argmax(Q[sid, legal])])
        best = next(a for a in legal if id_dist[sid ^ (1 << int(a))] == d0 - 1)
        flag = "" if pick == best else "   <- 學錯了"
        print(f"{d0:>4}  {''.join(map(str, s)):>11}  {txt:<34} 環{pick + 1}   環{best + 1}{flag}")
    print("\n注意每一列的兩個 Q 值只差 0.1 上下——決定行為的是「誰比較大」，不是絕對值。")
    print("DQN 那 138k 個參數做的事，就是用一個連續函數去逼近這張 512x9 的表。")


if __name__ == "__main__":
    main()
