#!/usr/bin/env python3
"""voice_watchdog.py — restart wedged/hung voice-pipeline containers.

Runs inside a container with the docker socket mounted; talks to the Docker
API over /var/run/docker.sock (no docker CLI, no root on the host).

Detects what docker's restart policy misses:
  1. voice-agent wedged: Running, CPU pinned above WEDGE_CPU_PCT and zero log
     output for LOG_SILENCE_MIN minutes, WEDGE_CHECKS consecutive minutes.
  2. A service Running but its HTTP health endpoint dead for HEALTH_FAILS
     consecutive minutes.
Rate limited: one restart per service per RESTART_COOLDOWN_S.

Env overrides: WATCHDOG_WEDGE_CPU_PCT, WATCHDOG_LOG_SILENCE_MIN,
WATCHDOG_HEALTH_FAILS, WATCHDOG_RESTART_COOLDOWN_S.
"""
import http.client
import json
import os
import time
import urllib.request

SOCK = "/var/run/docker.sock"
LOG_SILENCE_MIN = int(os.environ.get("WATCHDOG_LOG_SILENCE_MIN", "10"))
WEDGE_CPU_PCT = float(os.environ.get("WATCHDOG_WEDGE_CPU_PCT", "22"))
WEDGE_CHECKS = int(os.environ.get("WATCHDOG_WEDGE_CHECKS", "2"))
HEALTH_FAILS = int(os.environ.get("WATCHDOG_HEALTH_FAILS", "3"))
RESTART_COOLDOWN_S = int(os.environ.get("WATCHDOG_RESTART_COOLDOWN_S", "600"))
INTERVAL_S = 60

# service name -> health URL (checked only when the container exists and runs)
HEALTH = {
    "stt-server": "http://127.0.0.1:8080/health",
    "tts-voxcpm": "http://127.0.0.1:8402/health",
    "tts-adapter": "http://127.0.0.1:8399/health",
}

state = {}  # (name, kind) -> consecutive fail count; name -> last restart ts


def log(msg):
    print(f"[watchdog {time.strftime('%F %T')}] {msg}", flush=True)


class DockerSocket(http.client.HTTPConnection):
    def __init__(self):
        super().__init__("localhost")

    def connect(self):
        import socket
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(SOCK)


def docker(method, path):
    conn = DockerSocket()
    try:
        conn.request(method, path)
        resp = conn.getresponse()
        body = resp.read()
        return resp.status, body
    finally:
        conn.close()


def container_running(name):
    status, body = docker("GET", f"/containers/{name}/json")
    if status == 404:
        return None  # does not exist
    if status != 200:
        return None
    return json.loads(body)["State"].get("Running", False)


def restart(name, reason):
    now = time.time()
    last = state.get((name, "last_restart"), 0)
    if now - last < RESTART_COOLDOWN_S:
        log(f"SKIP {name} restart (cooldown, {int(now - last)}s ago): {reason}")
        return
    log(f"RESTART {name}: {reason}")
    status, body = docker("POST", f"/containers/{name}/restart")
    log(f"  restart http={status} {body[:120]!r}")
    state[(name, "last_restart")] = now
    state.pop((name, "wedge"), None)
    state.pop((name, "health"), None)


def cpu_pct(name):
    # Docker 26 one-shot stats arrive with empty precpu_stats and (on some
    # daemon versions) no system_cpu_usage, so percentages must be computed
    # from two consecutive frames of the streaming endpoint (~1 s apart).
    conn = DockerSocket()
    frames = []
    try:
        conn.request("GET", f"/containers/{name}/stats")
        resp = conn.getresponse()
        if resp.status != 200:
            return None
        dec = json.JSONDecoder()
        buf = b""
        deadline = time.time() + 8
        while len(frames) < 2 and time.time() < deadline:
            chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
            if not chunk:
                break
            buf += chunk
            while True:
                s = buf.lstrip()
                if not s:
                    buf = b""
                    break
                try:
                    obj, idx = dec.raw_decode(s.decode("utf-8", "ignore"))
                    frames.append(obj)
                    buf = s[idx:]
                except ValueError:
                    break
    except Exception:
        return None
    finally:
        conn.close()
    if len(frames) < 2:
        return None
    f1, f2 = frames
    try:
        cpu_delta = (f2["cpu_stats"]["cpu_usage"]["total_usage"]
                     - f1["cpu_stats"]["cpu_usage"]["total_usage"])
        sys_delta = (f2["cpu_stats"]["system_cpu_usage"]
                     - f1["cpu_stats"]["system_cpu_usage"])
        ncpu = f2["cpu_stats"].get("online_cpus") or 1
        if sys_delta <= 0:
            return None
        return cpu_delta / sys_delta * ncpu * 100.0
    except (KeyError, TypeError):
        return None


def log_lines_since(name, minutes):
    since = int(time.time()) - minutes * 60
    status, body = docker("GET", f"/containers/{name}/logs?stdout=1&stderr=1&since={since}")
    if status != 200:
        return None
    # docker log stream has an 8-byte frame header per record when not TTY
    return len(body.splitlines())


def healthy(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return b"ok" in r.read(200)
    except Exception:
        return False


def bump(key):
    n = state.get(key, 0) + 1
    state[key] = n
    return n


def clear(key):
    state.pop(key, None)


def tick():
    # 1. voice-agent wedge: pinned CPU + silent logs
    running = container_running("voice-agent")
    if running is False:
        log("voice-agent exists but is not running; starting it")
        docker("POST", "/containers/voice-agent/start")
    elif running:
        cpu = cpu_pct("voice-agent")
        silent = log_lines_since("voice-agent", LOG_SILENCE_MIN)
        if cpu is not None and silent is not None and cpu > WEDGE_CPU_PCT and silent == 0:
            n = bump(("voice-agent", "wedge"))
            log(f"voice-agent wedge signal {n}/{WEDGE_CHECKS}: {cpu:.1f}% CPU, no logs for {LOG_SILENCE_MIN}m")
            if n >= WEDGE_CHECKS:
                restart("voice-agent", f"wedged at {cpu:.1f}% CPU with silent logs")
        else:
            clear(("voice-agent", "wedge"))

    # 2. health endpoints
    for name, url in HEALTH.items():
        if not container_running(name):
            continue  # absent or deliberately stopped: not our business
        if healthy(url):
            clear((name, "health"))
            continue
        n = bump((name, "health"))
        log(f"health FAIL {name} {n}/{HEALTH_FAILS} ({url})")
        if n >= HEALTH_FAILS:
            restart(name, f"health endpoint dead for {n} consecutive minutes")


def main():
    log(f"watchdog up: wedge>{WEDGE_CPU_PCT}%cpu+silent{LOG_SILENCE_MIN}m x{WEDGE_CHECKS}, "
        f"health x{HEALTH_FAILS}, cooldown {RESTART_COOLDOWN_S}s")
    while True:
        try:
            tick()
        except Exception as e:
            log(f"tick error: {e!r}")
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
