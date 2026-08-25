"""九連環的規則核心 + 精確最優解。

這個檔案沒有任何機器學習，全是純數學。它存在的理由是：
這個謎題有「已知的最優解」，所以我們可以隨時知道 agent 離最優還差多少。
打磚塊沒有這種東西——那才是 RL 通常面對的狀況。

環的編號：ring 1 在最外側（可以自由上下），ring n 在最裡面。
狀態用長度 n 的 0/1 陣列 s 表示，s[i] = 1 代表第 i+1 環還在桿上。
目標：從全 1 變成全 0。

規則（只有兩條）：
    1. ring 1 隨時可以上或下。
    2. ring k（k >= 2）可以動，若且唯若 ring k-1 在桿上，而且 ring 1..k-2 全都不在。

推論：任何狀態下最多只有 2 個合法動作。也就是說整張狀態圖是一條「路徑」，
512 個狀態排成一直線，你只能往前或往後走一步。全上的狀態離終點 341 步。
"""
from __future__ import annotations

import numpy as np


def legal_mask(s) -> np.ndarray:
    """哪些環現在可以動。回傳長度 n 的 0/1 陣列。"""
    s = np.asarray(s)
    n = len(s)
    m = np.zeros(n, dtype=np.int8)
    m[0] = 1                      # 規則 1：最外環永遠能動
    for k in range(1, n):
        if s[k - 1] == 1 and (k == 1 or not s[: k - 1].any()):
            m[k] = 1              # 規則 2：至多命中一個 k
            break
    return m


def distance(s) -> int:
    """還要幾步才能全部解下來。這就是「格雷碼解碼」。

    把狀態當成格雷碼讀：b[n-1] = s[n-1]，b[i] = b[i+1] XOR s[i]，
    再把 b 當成一般二進位讀出來的整數，就是剩餘步數。

    為什麼會這樣？因為合法動作恰好讓這個整數 +1 或 -1，
    而終點（全 0）對應到 0。所以解謎 == 從 341 一路減到 0。
    """
    s = np.asarray(s)
    n = len(s)
    b = 0
    total = 0
    for i in range(n - 1, -1, -1):
        b ^= int(s[i])
        total |= b << i
    return total


def state_from_distance(d: int, n: int) -> np.ndarray:
    """distance() 的反函數：給定剩餘步數，還原出唯一的狀態。

    因為狀態圖是一條路徑，d 和狀態是一對一的。做 curriculum 時很好用：
    「從離終點 20 步的地方開始」就是 state_from_distance(20, n)。
    """
    b = [(d >> i) & 1 for i in range(n)]
    s = np.zeros(n, dtype=np.int8)
    for i in range(n):
        s[i] = b[i] ^ (b[i + 1] if i + 1 < n else 0)
    return s


def full_state(n: int) -> np.ndarray:
    return np.ones(n, dtype=np.int8)


def optimal_action(s) -> int:
    """最優動作：走那個讓 distance 變小的合法動作。

    也有閉式解（distance 是奇數就動 ring 1，偶數就動另一個合法環），
    但直接試最多兩個合法動作最不容易寫錯，速度也夠。
    """
    s = np.asarray(s)
    d0 = distance(s)
    if d0 == 0:
        return -1
    for i in np.flatnonzero(legal_mask(s)):
        t = s.copy()
        t[i] ^= 1
        if distance(t) == d0 - 1:
            return int(i)
    raise AssertionError("每個非終點狀態一定有一個往前的合法動作")


def optimal_moves(n: int = 9):
    """從全上走到全下的完整最優解，長度 (2^(n+1) - 1) / 3（n 為奇數）。"""
    s = full_state(n)
    out = []
    while distance(s) > 0:
        a = optimal_action(s)
        out.append(a)
        s[a] ^= 1
    return out


if __name__ == "__main__":
    for n in range(1, 12):
        print(f"n={n:2d}  最優步數 = {distance(full_state(n)):4d}")


# ---- 批次版本（給評估用，一次算 512 個狀態）----
# 內容跟上面單筆的版本完全等價，_smoke.py 會逐一比對過。

def legal_mask_batch(S: np.ndarray) -> np.ndarray:
    B, n = S.shape
    M = np.zeros((B, n), dtype=np.int8)
    M[:, 0] = 1
    csum = np.cumsum(S, axis=1)
    for k in range(1, n):
        below = csum[:, k - 2] if k >= 2 else np.zeros(B, dtype=csum.dtype)
        M[:, k] = (S[:, k - 1] == 1) & (below == 0)
    return M


def distance_batch(S: np.ndarray) -> np.ndarray:
    B, n = S.shape
    b = np.zeros(B, dtype=np.int64)
    total = np.zeros(B, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        b ^= S[:, i].astype(np.int64)
        total |= b << i
    return total


def all_states(n: int) -> np.ndarray:
    """512 個狀態，第 d 列就是離終點 d 步的那個狀態。"""
    return np.stack([state_from_distance(d, n) for d in range(2 ** n)])
