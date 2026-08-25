"""九連環的 numpy 環境（無畫面）。

規則跟 web/engine.js 一模一樣：兩邊都讀 shared/config.json，
所以 Python 訓練出來的權重，搬回瀏覽器可以直接用。

一個 step = 動一個環（上或下）。action = 環的索引 0..n-1，
不合法的動作會被 legal_mask() 擋掉（train.py 真的有擋）。

介面刻意做成 Gym 風格：reset() / step(action) -> (obs, reward, done, info)

這個環境和打磚塊最大的差別：
    * 完全沒有隨機性。同一串動作永遠得到同一個結果。
    * 獎勵極度稀疏——341 步裡只有最後一步有 +10。
    * 中途「一定要把已經拿下來的環裝回去」，貪心策略必死。
所以它是拿來練 reward shaping 和探索的，不是拿來練反應的。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from solver import distance, full_state, legal_mask, state_from_distance

CONFIG_PATH = Path(__file__).resolve().parent.parent / "shared" / "config.json"


def load_config(path=CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class RingsEnv:
    """start 有三種：
        "full"        每局都從全上開始（341 步）。這是真正要解的問題。
        "curriculum"  從離終點 <= k 步的地方開始，k 由外面慢慢調大。
        "random"      從 512 個狀態裡均勻挑一個。
    """

    def __init__(self, cfg=None, seed=None, start="full", start_max=None):
        self.cfg = cfg if cfg is not None else load_config()
        c = self.cfg
        self.n = int(c["rings"])
        self.max_steps = int(c["maxSteps"])
        self.gamma = float(c["gamma"])
        self.shaping_gamma = float(c.get("shapingGamma", 1.0))
        R = c["reward"]
        self.r_step = float(R["step"])
        self.r_solve = float(R["solve"])
        self.r_illegal = float(R["illegal"])
        self.r_shaping = float(R["shaping"])
        self.r_naive = float(R["naiveRingOff"])

        self.n_actions = self.n
        self.obs_size = 2 * self.n + 1
        self.full_distance = distance(full_state(self.n))  # n=9 -> 341
        self.start = start
        self.start_max = self.full_distance if start_max is None else int(start_max)
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ---- 狀態 ----

    def reset(self):
        if self.start == "full":
            self.s = full_state(self.n)
        elif self.start == "curriculum":
            d = int(self.rng.integers(1, max(2, self.start_max + 1)))
            self.s = state_from_distance(d, self.n)
        elif self.start == "random":
            self.s = state_from_distance(int(self.rng.integers(1, 2 ** self.n)), self.n)
        else:
            raise ValueError(self.start)
        self.dist = distance(self.s)
        self.start_dist = self.dist
        self.steps = 0
        self.illegal = 0
        self.over = False
        self.solved = False
        return self.obs()

    def legal(self) -> np.ndarray:
        return legal_mask(self.s)

    def obs(self) -> np.ndarray:
        """19 維：9 個環的狀態、9 個合法性、1 個時間進度。

        合法性其實可以從狀態算出來（是冗餘的），但餵進去讓網路少學一件事，
        而且 replay buffer 之後可以直接從 obs 切出 mask，不用另外存一份。
        """
        o = np.empty(self.obs_size, dtype=np.float32)
        o[: self.n] = self.s * 2.0 - 1.0          # 0/1 -> -1/+1
        o[self.n : 2 * self.n] = self.legal()
        o[2 * self.n] = self.steps / self.max_steps
        return o

    # ---- 互動 ----

    def step(self, action: int):
        assert not self.over, "這局已經結束了，先 reset()"
        a = int(action)
        self.steps += 1
        info = {}

        if not self.legal()[a]:
            # 不合法：狀態不動，只吃罰分。train.py 有做 masking，正常不會走到這裡。
            self.illegal += 1
            r = self.r_step + self.r_illegal
            self._maybe_timeout(info)
            return self.obs(), r, self.over, info

        removed = self.s[a] == 1
        self.s[a] ^= 1
        d_old, d_new = self.dist, distance(self.s)
        self.dist = d_new

        r = self.r_step
        # potential-based shaping（Ng et al. 1999）：F = shaping * (γ_s * Φ(s') - Φ(s))，Φ(s) = -distance(s)
        # 把「要走 341 步才拿得到分」變成「每走對一步就有回饋」。
        # shaping 設 0 就退回原本的稀疏獎勵——那樣幾乎學不起來，可以自己試（記得配 --gamma 0.999）。
        #
        # 這裡有一個真的會踩到的坑，值得看清楚：
        # 理論保證「最優策略不變」要求 γ_s 等於訓練用的 gamma。但 Φ 和距離成正比，
        # 展開後會多出一項 (1-γ_s) * d。d 最大 341，所以 γ_s=0.95 時這一項就有 17，
        # 是真正帶方向的 ±1 的十七倍——結果是「走錯也拿正分」，agent 完全學不到方向。
        # 這不是推測，ml/ablation.py 跑得出來：教科書寫法最後是 0.0%，這裡的寫法是 100.0%。
        # 所以 shapingGamma 預設 1.0，讓 F 剛好等於 ±shaping。
        # 代價是理論上不再保證最優策略不變；但 step cost 已經把「越短越好」寫進去了，
        # 實務上結果就是最優解（evaluate.py 會逐一驗證 512 個起點）。
        r += self.r_shaping * (self.shaping_gamma * (-d_new) - (-d_old))
        # 這一項預設是 0。設成正數就是「拿下一個環 +x、裝回去 -x」的直覺獎勵，
        # 也就是一個會主動把 agent 教壞的 reward——九連環一定要退才能進。
        r += self.r_naive * (1.0 if removed else -1.0)

        if d_new == 0:
            self.solved = True
            self.over = True
            r += self.r_solve
            info["solved"] = True
        else:
            self._maybe_timeout(info)

        if self.over:
            info.setdefault("solved", self.solved)
            info["ep_steps"] = self.steps
            info["ep_dist"] = self.dist
            info["ep_start_dist"] = self.start_dist
            info["ep_illegal"] = self.illegal
        return self.obs(), r, self.over, info

    def _maybe_timeout(self, info):
        if self.steps >= self.max_steps:
            self.over = True
            info["timeout"] = True
            info["ep_steps"] = self.steps
            info["ep_dist"] = self.dist
            info["ep_start_dist"] = self.start_dist
            info["ep_illegal"] = self.illegal


class VecRings:
    """一次跑很多局。這個環境單步超便宜，但 GPU 推論要湊 batch 才划算。"""

    def __init__(self, n_envs: int, cfg=None, seed=0, start="full", start_max=None):
        self.cfg = cfg if cfg is not None else load_config()
        self.envs = [RingsEnv(self.cfg, seed=seed + i, start=start, start_max=start_max)
                     for i in range(n_envs)]
        self.n_envs = n_envs
        self.obs_size = self.envs[0].obs_size
        self.n_actions = self.envs[0].n_actions

    def set_start_max(self, k: int):
        for e in self.envs:
            e.start_max = int(k)

    def reset(self):
        return np.stack([e.reset() for e in self.envs])

    def legal(self):
        return np.stack([e.legal() for e in self.envs]).astype(np.float32)

    def step(self, actions):
        obs = np.empty((self.n_envs, self.obs_size), dtype=np.float32)
        rew = np.empty(self.n_envs, dtype=np.float32)
        done = np.zeros(self.n_envs, dtype=bool)
        infos = [None] * self.n_envs
        for i, e in enumerate(self.envs):
            o, r, d, info = e.step(int(actions[i]))
            if d:
                o = e.reset()     # 自動開下一局；done=1 時 target 不會用到 s2，所以不影響正確性
            obs[i], rew[i], done[i], infos[i] = o, r, d, info
        return obs, rew, done, infos


def mask_from_obs(obs):
    """從 obs 切出合法動作遮罩。obs 的第 n..2n-1 維就是 mask，所以不用另外存。"""
    n = (obs.shape[-1] - 1) // 2
    return obs[..., n : 2 * n]
