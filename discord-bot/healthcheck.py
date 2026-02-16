"""
Lightweight HTTP health-check server for AWS App Runner.

App Runner requires a service that listens on a TCP port and responds to
HTTP health checks. This module runs a minimal HTTP server in a background
thread alongside the Discord bot.
"""

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

# Populated by bot.py at startup
_bot_ref = None
_qa_ref = None
_start_time = time.time()


class HealthHandler(BaseHTTPRequestHandler):
    """Handles health check and status requests."""

    def do_GET(self):
        if self.path == "/health" or self.path == "/":
            self._respond_health()
        elif self.path == "/status":
            self._respond_status()
        else:
            self.send_response(404)
            self.end_headers()

    def _respond_health(self):
        """Simple health check — returns 200 if the process is alive."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        body = json.dumps({"status": "healthy"})
        self.wfile.write(body.encode())

    def _respond_status(self):
        """Detailed status including bot and QA engine state."""
        bot = _bot_ref
        qa = _qa_ref() if callable(_qa_ref) else _qa_ref

        bot_ready = bot is not None and bot.is_ready() if bot else False
        qa_ready = qa is not None
        uptime_s = int(time.time() - _start_time)

        status = {
            "status": "healthy",
            "bot_connected": bot_ready,
            "bot_user": str(bot.user) if bot and bot_ready else None,
            "bot_guilds": len(bot.guilds) if bot and bot_ready else 0,
            "qa_engine_ready": qa_ready,
            "qa_doc_chunks": len(qa.chunks) if qa_ready else 0,
            "uptime_seconds": uptime_s,
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())

    def log_message(self, format, *args):
        """Suppress default request logging to keep logs clean."""
        pass


def start_health_server(port: int = 8080, bot_ref=None, qa_ref=None):
    """
    Start the health-check HTTP server in a daemon thread.

    Args:
        port: TCP port to listen on (App Runner uses PORT env var, default 8080).
        bot_ref: Reference to the discord.py Bot instance.
        qa_ref: Callable that returns the QAEngine instance (lambda for lazy access).
    """
    global _bot_ref, _qa_ref
    _bot_ref = bot_ref
    _qa_ref = qa_ref

    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[healthcheck] Listening on port {port}")
