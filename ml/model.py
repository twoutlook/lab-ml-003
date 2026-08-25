"""Q 網路（Dueling MLP）與 n-step replay buffer。

跟 lab-ml-001 幾乎一樣，只有一個關鍵差別：這裡的動作有合法 / 不合法之分，
所以多了 masked_q()。取 argmax 前要先把不合法的動作壓成 -inf，
不然網路會把大量容量浪費在「學會不要選那些根本走不動的環」上。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

NEG_INF = -1e9


class DuelingQNet(nn.Module):
    """Dueling DQN：把 Q 拆成「這個狀態本身好不好(V)」+「這個動作比平均好多少(A)」。

    Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)

    在九連環特別有用：每個狀態最多只有 2 個動作是能走的，
    V(s) 幾乎就等於「離終點還有多遠」，那是網路最該先學會的東西。
    """

    def __init__(self, obs_size: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.adv = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        v = self.value(h)
        a = self.adv(h)
        return v + a - a.mean(dim=1, keepdim=True)


def masked_q(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """把不合法動作的 Q 值壓到 -inf。mask 是 1 = 合法。"""
    return q.masked_fill(mask < 0.5, NEG_INF)


class ReplayBuffer:
    """均勻取樣的經驗回放池。

    存的是「n-step transition」：(s_t, a_t, R_t^{(n)}, s_{t+n}, done)
    其中 R_t^{(n)} = r_t + gamma*r_{t+1} + ... + gamma^{n-1}*r_{t+n-1}

    n-step 的用意：讓獎勵訊號一次往回傳 n 步，學得比 1-step 快很多。
    在 341 步長的任務上，這個差別特別明顯。

    注意這裡沒有另外存 s2 的合法遮罩——因為 obs 本身就含著它（env.mask_from_obs）。
    """

    def __init__(self, capacity: int, obs_size: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.s = np.zeros((capacity, obs_size), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_size), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        # 實際累積了幾步（一局結尾可能不足 n），算 target 時 gamma 要開對次方
        self.k = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def add(self, s, a, r, s2, d, k):
        i = self.idx
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = s2
        self.d[i] = d
        self.k[i] = k
        self.idx += 1
        if self.idx >= self.capacity:
            self.idx = 0
            self.full = True

    def sample(self, batch_size: int, rng: np.random.Generator):
        n = len(self)
        i = rng.integers(0, n, size=batch_size)
        t = lambda x: torch.as_tensor(x, device=self.device)
        return t(self.s[i]), t(self.a[i]), t(self.r[i]), t(self.s2[i]), t(self.d[i]), t(self.k[i])


class NStepAccumulator:
    """每個 env 各自維護一條 n-step 佇列，湊滿 n 步（或這局結束）就吐出一筆 transition。"""

    def __init__(self, n_envs: int, n_step: int, gamma: float):
        self.n_step = n_step
        self.gamma = gamma
        self.buf = [[] for _ in range(n_envs)]

    def push(self, env_i: int, s, a, r, s2, done):
        q = self.buf[env_i]
        q.append((s, a, r, s2, done))
        out = []
        if len(q) >= self.n_step:
            out.append(self._make(q))
            q.pop(0)
        if done:
            # 這局結束了，把佇列裡剩下的短 transition 全部倒出來（不跨局）
            while q:
                out.append(self._make(q))
                q.pop(0)
        return out

    def _make(self, q):
        s, a = q[0][0], q[0][1]
        R = 0.0
        for k, (_, _, r, _, _) in enumerate(q):
            R += (self.gamma ** k) * r
        s2, done = q[-1][3], q[-1][4]
        return (s, a, R, s2, float(done), len(q))
