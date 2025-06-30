from __future__ import annotations
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from .manager import MultiAgentManager


class MCPRequestHandler(BaseHTTPRequestHandler):
    manager = MultiAgentManager()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/assist":
            params = parse_qs(parsed.query)
            agent = params.get("agent", ["operation"])[0]
            query = params.get("q", [""])[0]
            response = self.manager.assist(agent, query)
            self._send_json({"response": response})
        else:
            self.send_response(404)
            self.end_headers()

    def _send_json(self, data) -> None:
        payload = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class MCPServer:
    """Tiny HTTP server exposing agent assistance."""

    def __init__(self, host: str = "localhost", port: int = 8000) -> None:
        self.server = HTTPServer((host, port), MCPRequestHandler)

    def serve_forever(self) -> None:
        print(f"MCP server running on {self.server.server_address}")
        self.server.serve_forever()
