"""建立（或找到）playlist、組出描述、上傳影片，然後把影片網址寫回 out/youtube.json。

    python tools/publish_youtube.py --dry-run              # 只印出標題與描述，不上傳
    python tools/publish_youtube.py                        # 上傳（預設 unlisted）
    python tools/publish_youtube.py --privacy public       # 看過之後再公開
    python tools/publish_youtube.py --update <videoId>     # 只改描述

playlist 一定要在上傳「之前」就存在：描述裡必須帶自己 playlist 的網址，
而共用的 uploader 是上傳完才建 playlist，所以這裡先建、拿到真網址再寫描述。

章節時間直接從 out/plan.json 讀——那是錄這支影片時實際用的分鏡表，
不是人工抄的，改了影片長度也不會對不上。
結果數字從 ml/checkpoints/benchmark.json 和 ablation.json 讀，
所以描述裡的百分比不可能跟實際跑出來的不一樣。

配額：videos.insert 一次約 1,600 units，每天上限 10,000。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import time

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN = os.environ.get(
    "YT_TOKEN", r"C:\Users\mark\Documents\2026-mark-locally-only\yt_token.json")
UPLOADER = r"C:\2026BizProject\GOAL\001\routine\upload_youtube.py"
SCOPES = ["https://www.googleapis.com/auth/youtube"]

PLAYLIST = "ai-ml-lab"
PLAYLIST_DESC = (
    "AI / ML lab — 從零手寫的機器學習練習：自己寫環境、自己寫演算法、自己訓練。"
    "Hand-written machine-learning experiments: own environment, own algorithm, own training loop."
)

# Claude artifact 預設是私人的，擁有者去分享之後外人才打得開。
# 公開影片連到一個沒分享的 artifact，觀眾只會看到一面牆。
ARTIFACT_URL = "https://claude.ai/code/artifact/ab8383b4-9e64-47ae-8578-800957e11865"
REPO_URL = "https://github.com/twoutlook/lab-ml-003"

VIDEO = os.path.join(ROOT, "out", "rings-dqn.mp4")
PLAN = os.path.join(ROOT, "out", "plan.json")
BENCH = os.path.join(ROOT, "ml", "checkpoints", "benchmark.json")
ABL = os.path.join(ROOT, "ml", "checkpoints", "ablation.json")
OUT_JSON = os.path.join(ROOT, "out", "youtube.json")
DESC_PATH = os.path.join(ROOT, "out", "youtube-desc.txt")

TITLE = "讓程式自己學會解九連環｜341 步最優解，以及一個照教科書寫就會失敗的 reward shaping"
TAGS = ("reinforcement learning,DQN,Double DQN,reward shaping,強化學習,機器學習,深度學習,"
        "九連環,Chinese rings,baguenaudier,格雷碼,Gray code,PyTorch,numpy,Claude Code,RL from scratch")

CHAPTER_NAMES = {
    "title": "九連環為什麼適合拿來練 RL",
    "rules": "規則只有兩條，但足以逼你後退",
    "greedy": "直覺策略：能拿就拿，然後卡死",
    "random": "地板：完全亂走",
    "dqn": "DQN agent — 18 個 0 跟 1，還有動作遮罩",
    "shaping": "踩到的坑：照教科書寫 shaping 會學不起來",
    "results": "窮舉 511 個起點的結果",
}


def creds():
    c = Credentials.from_authorized_user_file(TOKEN, SCOPES)
    if not c.valid:
        if c.expired and c.refresh_token:
            c.refresh(Request())
            bak = TOKEN + time.strftime(".bak-%Y%m%d-%H%M%S")
            try:
                with open(TOKEN) as f, open(bak, "w") as g:
                    g.write(f.read())
            except OSError:
                pass
            with open(TOKEN, "w") as f:
                f.write(c.to_json())
            print(f"token refreshed (backup: {os.path.basename(bak)})")
        else:
            sys.exit("token invalid and cannot refresh — 先跑 ai-drama-v2/yt_auth.py")
    return c


def find_or_create(yt, title, privacy):
    req = yt.playlists().list(part="snippet", mine=True, maxResults=50)
    while req is not None:
        res = req.execute()
        for it in res.get("items", []):
            if it["snippet"]["title"].strip().lower() == title.lower():
                print(f"playlist exists: {title} ({it['id']})")
                return it["id"]
        req = yt.playlists().list_next(req, res)
    pl = yt.playlists().insert(
        part="snippet,status",
        body={"snippet": {"title": title, "description": PLAYLIST_DESC},
              "status": {"privacyStatus": "public" if privacy == "public" else "unlisted"}},
    ).execute()
    print(f"playlist created: {title} ({pl['id']})")
    return pl["id"]


def mmss(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def chapters():
    """章節時間來自這支影片自己的分鏡表，不是手抄的。"""
    if not os.path.exists(PLAN):
        raise SystemExit(f"缺 {PLAN} — 章節時間必須來自實際錄製的那一次")
    with io.open(PLAN, encoding="utf-8") as f:
        plan = json.load(f)
    fps = plan["fps"]
    # YouTube 的章節第一條一定要是 00:00
    return "\n".join(f"{mmss(s['start'] / fps)}  {CHAPTER_NAMES.get(s['id'], s['id'])}"
                     for s in plan["scenes"])


def results_table():
    with io.open(BENCH, encoding="utf-8") as f:
        b = json.load(f)
    out = [f"（從全部 {b['starts']} 個非終點狀態各解一次。環境是決定性的，這是窮舉不是抽樣）", ""]
    for r in b["rows"]:
        full = f" {r['full_steps']} 步" if r["full_steps"] else "：沒解開"
        out.append(f"・{r['name']}：解開 {r['solve_rate'] * 100:.1f}%"
                   f" / 剛好最優 {r['optimal_rate'] * 100:.1f}%"
                   f" / 從全上出發{full}")
    return "\n".join(out)


def ablation_table():
    with io.open(ABL, encoding="utf-8") as f:
        a = json.load(f)
    out = [f"（同一份程式、同一組超參數，只改 shaping 的一個常數，各跑 {a['steps']:,} 個 transition）", ""]
    for r in a["runs"]:
        out.append(f"・{r['name']} — {r['setting']}：最後走出最優的起點比例 {r['final_optimal_rate'] * 100:.1f}%")
    return "\n".join(out)


def build_description(playlist_url):
    return f"""九連環是中國最古老的益智玩具之一，把九個環全部解下來最少要三百四十一步。
這個數字不是估的，是用格雷碼算出來的——所以當我們讓程式自己學著解它，
可以精確地知道它離最優解差了幾步。這是強化學習裡很少有的奢侈品。

謎題、環境、神經網路全部手寫，沒有用 Gym，也沒有用 Stable-Baselines3。

▶ 播放清單 / Playlist: {playlist_url}
▶ 原始碼 / Source: {REPO_URL}
▶ 可互動的完整圖文版（中英雙語，網頁上的 agent 是真的在跑推論）/ Interactive write-up: {ARTIFACT_URL}

{chapters()}

── 結果 ──
{results_table()}

「能拿下來就拿下來」這個直覺策略，在 511 個起點裡只解得開 1 個。
因為九連環的 341 步裡有一半是在把已經解下來的環裝回去——不肯後退就走不完。

── 影片的重點其實是這個坑 ──
{ablation_table()}

341 步只有最後一步有分，這叫稀疏獎勵，標準解法是 potential-based reward shaping：
    F(s,s') = shaping × ( γs · Φ(s') − Φ(s) )，Φ(s) = −distance(s)

Ng et al. (1999) 的保證要求 γs 等於訓練用的折扣率，這樣最優策略才不會被改掉。
照做的結果是完全學不起來。原因在展開之後：Φ 跟距離成正比，會多出一項 (1 − γs) × d，
d 最大 341，γs = 0.95 時這一項有 17，是真正帶方向的那個 ±1 的十七倍——走錯邊也拿正分。
把 γs 設成 1 才學得起來，代價是理論保證消失，所以才必須窮舉 511 個起點去驗證。

理論保證的前提沒滿足的時候，要知道它為什麼沒滿足。

── 怎麼做的 ──
・環境：numpy 手寫，跟瀏覽器版共用同一份 shared/config.json
・演算法：Double DQN + Dueling head + 5-step return，137,994 個參數
・狀態：19 維 —— 9 個環在不在桿上、9 個合法動作遮罩、1 個時間進度
・動作：9 個（動第 i 個環）。不合法的在選動作和算 target 時都壓成 −1e9
・訓練：800,000 個 transition，RTX 4070 Ti SUPER 上約 3 分鐘
・部署：權重匯出成 JSON，瀏覽器用純 JS 做前向傳播，不需要 TensorFlow.js
・對帳：因為環境沒有隨機性，JS 端跟 Python 端要求逐位相同，差一步就是 bug
・語音：edge-tts zh-TW-HsiaoChenNeural
・影片：headless Chrome 逐格離線算圖（非螢幕錄影，不會掉格），旁白時間軸由程式計算對齊

── 也照實說一件事 ──
這個問題其實不需要神經網路。512 個狀態直接開一張 Q 表，表格式 Q-learning 五千局
就收斂到 100% 最優，比 DQN 快也比 DQN 準。這個專案值得學的是方法——動作遮罩、
reward shaping、窮舉式評估——不是結果。

── English ──
A from-scratch reinforcement-learning project on the Chinese rings puzzle (baguenaudier).
Solving all nine rings takes exactly 341 moves, computable in closed form from a Gray code,
so the agent's gap to optimal is measurable rather than estimated. Everything is hand-written:
the puzzle, the numpy environment, the Dueling Double-DQN, the browser inference.
Evaluation enumerates all 511 non-terminal start states — no sampling, no seeds.
The trained agent solves 100% of them in exactly the optimal number of moves.
The real subject of the video is a reward-shaping trap: following the textbook condition
γs = γ makes the task unlearnable, because the potential is proportional to distance and the
leftover (1 − γs)·d term is seventeen times the directional signal. Full bilingual write-up,
with the network running live in the page, at the link above.

Created by MarkLuce AI · Claude Code · Claude Opus 5
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--privacy", default="unlisted", choices=["unlisted", "public", "private"],
                    help="預設 unlisted：先傳上去看過，再決定要不要公開")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--update", help="只更新這個 videoId 的描述")
    a = ap.parse_args()

    yt = build("youtube", "v3", credentials=creds())
    pid = find_or_create(yt, PLAYLIST, a.privacy)
    purl = f"https://www.youtube.com/playlist?list={pid}"
    print(f"PLAYLIST: {purl}")

    desc = build_description(purl)
    with io.open(DESC_PATH, "w", encoding="utf-8") as f:
        f.write(desc)

    if a.dry_run:
        print(f"\nTITLE ({len(TITLE)} chars): {TITLE}\n")
        print("-" * 70)
        print(desc)
        print("-" * 70)
        print(f"\ndry run：不會上傳。描述已寫到 {DESC_PATH}")
        return

    if a.update:
        cur = yt.videos().list(part="snippet", id=a.update).execute()["items"][0]["snippet"]
        cur["description"] = desc
        yt.videos().update(part="snippet", body={"id": a.update, "snippet": cur}).execute()
        print(f"description updated on {a.update}")
        return

    if not os.path.exists(VIDEO):
        sys.exit(f"找不到影片 {VIDEO}")
    size_mb = os.path.getsize(VIDEO) / 1048576
    print(f"uploading {os.path.basename(VIDEO)} — {size_mb:.0f} MB as {a.privacy} …")
    # videoId 直接從 uploader 的 stdout 撈：playlistItems 有傳播延遲，
    # 剛上傳的影片常常還反查不到。
    proc = subprocess.run(
        [sys.executable, UPLOADER,
         "--video", VIDEO, "--title", TITLE, "--desc-file", DESC_PATH,
         "--playlist", PLAYLIST, "--privacy", a.privacy, "--tags", TAGS],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    m = re.search(r"uploaded video:\s*([\w-]{11})", proc.stdout) or \
        re.search(r"youtu\.be/([\w-]{11})", proc.stdout)
    vid = m.group(1) if m else None
    info = {"playlist_id": pid, "playlist_url": purl,
            "video_id": vid, "video_url": f"https://youtu.be/{vid}" if vid else None,
            "title": TITLE, "privacy": a.privacy}
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if not vid:
        print("警告：videoId 沒抓到。artifact 的反向連結需要它——請手動補進 out/youtube.json。")
    else:
        print("\n接著跑：node tools/build_artifact.mjs   把影片連結灌回 artifact")


if __name__ == "__main__":
    main()
