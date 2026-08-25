"""五秒鐘檢查：規則、批次版本、環境、shaping 有沒有壞掉。

    python _smoke.py

改完 solver.py / env.py 一定要跑這個。它擋掉的是「訓練跑得動、
但學出來的東西搬到網頁上就爛掉」那一類最難查的 bug。
"""
from __future__ import annotations

import numpy as np

from env import RingsEnv, VecRings, load_config, mask_from_obs
from solver import (all_states, distance, distance_batch, full_state, legal_mask,
                    legal_mask_batch, optimal_action, optimal_moves, state_from_distance)

cfg = load_config()
n = int(cfg["rings"])

# 1. 單筆和批次版本必須逐位相同
S = all_states(n)
assert (legal_mask_batch(S) == np.stack([legal_mask(s) for s in S])).all(), "legal_mask 批次版不一致"
assert (distance_batch(S) == np.array([distance(s) for s in S])).all(), "distance 批次版不一致"

# 2. distance 和 state 必須一對一，且全上的距離要等於 (2^(n+1)-1)/3
assert len(set(distance_batch(S).tolist())) == 2 ** n, "distance 不是一對一"
expect = (2 ** (n + 1) - 1) // 3 if n % 2 else (2 ** (n + 1) - 2) // 3
assert distance(full_state(n)) == expect, f"全上距離應為 {expect}"

# 3. 每個非終點狀態都必須恰好有一個「往前」的合法動作，而且合法動作最多 2 個
for d in range(1, 2 ** n):
    s = state_from_distance(d, n)
    m = legal_mask(s)
    assert 1 <= m.sum() <= 2, "合法動作應該是 1 或 2 個"
    a = optimal_action(s)
    t = s.copy(); t[a] ^= 1
    assert distance(t) == d - 1

# 4. 最優解真的走得完，而且步數等於距離
moves = optimal_moves(n)
assert len(moves) == expect, "最優解長度不對"

# 5. 環境：照最優解走，必須剛好 341 步解開，沒有不合法動作
e = RingsEnv(cfg)
total = 0.0
for a in moves:
    _, r, done, info = e.step(a)
    total += r
assert e.solved and e.steps == expect and e.illegal == 0, "env 跑最優解沒解開"

# 6. shaping 的方向要對：走對加分、走錯扣分（這是最容易調壞的地方）
e = RingsEnv(cfg)
good = e.step(optimal_action(e.s))[1]
e2 = RingsEnv(cfg)
bad_a = [i for i in np.flatnonzero(e2.legal()) if i != optimal_action(e2.s)]
bad = e2.step(bad_a[0])[1] if bad_a else None
assert good > 0 > bad, f"shaping 方向錯了：走對 {good:+.3f}，走錯 {bad:+.3f}"

# 7. obs 裡切出來的 mask 要等於 env 的 legal()
v = VecRings(8, cfg=cfg, seed=1, start="random")
obs = v.reset()
assert (mask_from_obs(obs) == v.legal()).all(), "mask_from_obs 對不上"

print(f"全部通過。n={n}  最優步數={expect}  obs_size={e.obs_size}")
print(f"  照最優解走一遍的總 reward = {total:.3f}")
print(f"  走對一步 {good:+.3f}，走錯一步 {bad:+.3f}")
