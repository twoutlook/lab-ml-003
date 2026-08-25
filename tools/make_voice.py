"""用 edge-tts 把 tools/script.json 的每一段旁白轉成 mp3，並量出長度。

    python tools/make_voice.py

輸出：
    out/voice/<id>.mp3
    out/voice/timing.json   每一段的長度（秒），錄影用它決定每個場景要幾張 frame
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import edge_tts

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "script.json"
OUT = ROOT / "out" / "voice"


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return round(float(out.stdout.strip()), 3)


async def main():
    spec = json.loads(SCRIPT.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    timing = []
    for sc in spec["scenes"]:
        mp3 = OUT / f"{sc['id']}.mp3"
        comm = edge_tts.Communicate(sc["text"], spec["voice"], rate=spec.get("rate", "+0%"))
        await comm.save(str(mp3))
        d = probe_duration(mp3)
        timing.append({"id": sc["id"], "duration": d, "chars": len(sc["text"])})
        print(f"{sc['id']:<10} {d:6.2f}s  {len(sc['text']):>3} 字")
    total = sum(t["duration"] for t in timing)
    (OUT / "timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n旁白總長 {total:.1f}s ({total/60:.2f} 分)")
    print(f"寫出 {OUT / 'timing.json'}")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
