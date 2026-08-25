"""本機預覽用：跟 python -m http.server 一樣，但 .html 會帶 charset=utf-8。
Artifact 正式發布時外層 wrapper 本來就會給 UTF-8，這支只是為了本機看得到正確的字。
    python tools/serve_utf8.py 8010
"""
import functools, http.server, socketserver, sys

class H(http.server.SimpleHTTPRequestHandler):
    def guess_type(self, path):
        t = super().guess_type(path)
        if t and t.startswith("text/html"):
            return "text/html; charset=utf-8"
        return t

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8010
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", port), H) as srv:
    print(f"serving on http://127.0.0.1:{port}")
    srv.serve_forever()
