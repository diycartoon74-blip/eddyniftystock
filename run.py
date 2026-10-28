"""Launch the NiftyPulse dashboard server."""

from __future__ import annotations

import socket
import sys

import uvicorn

from app.config import Settings


def _port_in_use(host: str, port: int) -> bool:
    """True when something (probably another NiftyPulse) already owns the port."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        infos = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (host, port))]
    for family, socktype, proto, _, addr in infos:
        with socket.socket(family, socktype, proto) as sock:
            try:
                sock.bind(addr)
            except OSError:
                return True
    return False


def main() -> None:
    settings = Settings()
    url = f"http://{settings.host}:{settings.port}"

    if _port_in_use(settings.host, settings.port):
        print("")
        print(f"  Port {settings.port} is already in use - NiftyPulse is probably running.")
        print(f"  Open {url} in your browser instead of starting a second copy.")
        print(f"  To restart, stop the process on port {settings.port}, then run this again.")
        print("")
        sys.exit(1)

    from app.server import app

    print("")
    print("  NiftyPulse - NIFTY / BANKNIFTY / SENSEX options (Made by EDDY)")
    print(f"  Dashboard: {url}")
    print(f"  Mode: {'DEMO' if settings.demo else 'LIVE data'} | "
          f"Broker: {settings.broker} | Auto-trade: {settings.auto_trade}")
    print("  Educational software. Not investment advice.")
    print("")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
