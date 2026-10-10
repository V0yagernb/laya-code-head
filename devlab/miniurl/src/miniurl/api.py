"""极简 HTTP 层（只为演示，不追求生产可用）。"""

import json
from http.server import BaseHTTPRequestHandler

from .shorten import expand, shorten


class Handler(BaseHTTPRequestHandler):
    store = None  # 由 create_server 注入

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        code = shorten(payload["url"], self.store)
        self._send(200, {"code": code})

    def do_GET(self):
        code = self.path.strip("/")
        try:
            url = expand(code, self.store)
        except KeyError:
            self._send(404, {"error": "unknown code"})
            return
        self._send(302, {"url": url})

    def _send(self, status: int, body: dict):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
