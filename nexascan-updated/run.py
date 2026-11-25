"""Start DiffSight on a rarely used port without touching anything else on the machine.

Port: NEXASCAN_PORT (default 38219). If it is busy the next free port in 38200-38229 is used; nothing is
ever killed or reused. Host: 127.0.0.1 unless NEXASCAN_HOST is set (for example 0.0.0.0).
Access log is off unless NEXASCAN_LOG_ACCESS=1.
"""
import os
import socket
import sys

DEFAULT_PORT = 38219
RANGE = range(38200, 38230)


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def pick_port(host: str) -> int:
    want = int(os.environ.get("NEXASCAN_PORT", DEFAULT_PORT))
    for p in [want] + [p for p in RANGE if p != want]:
        if port_free(host, p):
            return p
    print(f"No free port in {RANGE.start}-{RANGE.stop - 1}. Set NEXASCAN_PORT to another free port.", file=sys.stderr)
    sys.exit(1)


def log_config() -> dict:
    """uvicorn's default console logging plus a file handler: server messages also go to logs/nexascan.log."""
    import copy

    from uvicorn.config import LOGGING_CONFIG

    from nexascan import applog

    lc = copy.deepcopy(LOGGING_CONFIG)
    lc["formatters"]["plain"] = {"format": applog.FMT}
    lc["handlers"]["file"] = {"class": "logging.FileHandler", "filename": str(applog.LOG_FILE), "formatter": "plain", "encoding": "utf-8"}
    for name in ("uvicorn", "uvicorn.access"):
        lc["loggers"].setdefault(name, {}).setdefault("handlers", []).append("file")
    return lc


if __name__ == "__main__":
    import uvicorn

    from nexascan import applog
    applog.trim_at_startup()
    log = applog.get_logger()

    host = os.environ.get("NEXASCAN_HOST", "127.0.0.1")
    port = pick_port(host)
    if str(port) != os.environ.get("NEXASCAN_PORT", str(DEFAULT_PORT)):
        print(f"Port busy, using {port} instead.")
        log.warning("requested port was busy; using %s instead", port)
    log.info("starting DiffSight on %s:%s (Python %s, log file %s)", host, port, sys.version.split()[0], applog.LOG_FILE)
    print(f"DiffSight: http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
    uvicorn.run("nexascan.api:app", host=host, port=port, reload=False,
                access_log=os.environ.get("NEXASCAN_LOG_ACCESS") == "1", log_config=log_config())
