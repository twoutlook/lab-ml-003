import io, json, os
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

TOKEN = os.environ.get("YT_TOKEN", r"C:\Users\mark\Documents\2026-mark-locally-only\yt_token.json")
info = json.load(io.open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "youtube.json"), encoding="utf-8"))
yt = build("youtube", "v3", credentials=Credentials.from_authorized_user_file(TOKEN, ["https://www.googleapis.com/auth/youtube"]))

v = yt.videos().list(part="snippet,status,contentDetails", id=info["video_id"]).execute()["items"][0]
d = v["snippet"]["description"]
print("title     ", v["snippet"]["title"])
print("privacy   ", v["status"]["privacyStatus"], "| upload:", v["status"]["uploadStatus"])
print("duration  ", v["contentDetails"]["duration"])
print("tags      ", len(v["snippet"].get("tags", [])))
print("playlist link in desc:", info["playlist_id"] in d)
print("artifact link in desc:", "ab8383b4" in d)
print("chapters start at 00:00:", d.lstrip().count("00:00") > 0 or "\n00:00" in d)

items = yt.playlistItems().list(part="snippet", playlistId=info["playlist_id"], maxResults=10).execute()["items"]
print("playlist items:", [(i["snippet"]["resourceId"]["videoId"], i["snippet"]["title"][:20]) for i in items])
pl = yt.playlists().list(part="snippet,status", id=info["playlist_id"]).execute()["items"][0]
print("playlist  ", pl["snippet"]["title"], "|", pl["status"]["privacyStatus"])
