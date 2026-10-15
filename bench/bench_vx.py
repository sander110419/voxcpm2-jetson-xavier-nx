#!/usr/bin/env python3
"""VoxCPM2 bench — accurate wall + TTFA against audiocpp_server :8402 (stdlib only)."""
import json, http.client, time, sys, os, wave, struct

LABEL = sys.argv[1] if len(sys.argv) > 1 else "base"
OUT = "/tmp/vxbench"
os.makedirs(OUT, exist_ok=True)
SHORT = "Hoi. Dit is een test."
PARA = ("Goedemorgen. Vandaag wordt een drukke dag, dus ik heb alvast een paar "
        "dingen voor je uitgezocht. Het weer wordt zacht, met een graad of vijftien en af "
        "en toe een bui. Op de kalender staat vanavond een etentje, en morgenochtend is er "
        "ruimte voor die vergadering. Zal ik daar alvast een korte samenvatting voor "
        "voorbereiden?")
RES = open(os.path.join(OUT, LABEL + "_results.txt"), "w")

def req(text, stream, out_path):
    body = json.dumps({"model": "voxcpm2", "input": text, "stream": stream,
                       "stream_format": "audio", "response_format": "pcm"}).encode()
    conn = http.client.HTTPConnection("127.0.0.1", 8402, timeout=600)
    t0 = time.time()
    conn.request("POST", "/v1/audio/speech", body=body,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    assert resp.status == 200, resp.read()[:300]
    first_audio = None
    total = 0
    with open(out_path, "wb") as f:
        while True:
            chunk = resp.read1(65536)
            if not chunk:
                break
            if first_audio is None and chunk:
                first_audio = time.time() - t0
            f.write(chunk)
            total += len(chunk)
    wall = time.time() - t0
    conn.close()
    dur = total / 2 / 48000.0
    return wall, first_audio, dur, total

def log(line):
    print(line, flush=True)
    RES.write(line + "\n"); RES.flush()

# warmup
req("warm", True, "/dev/null")
for i in (1, 2, 3):
    w, fa, dur, n = req(SHORT, True, f"{OUT}/{LABEL}_short{i}.pcm")
    log(f"short{i} stream wall={w:.2f}s ttfa={fa:.2f}s audio={dur:.2f}s bytes={n}")
w, fa, dur, n = req(SHORT, False, f"{OUT}/{LABEL}_short_buffered.pcm")
log(f"short_buffered wall={w:.2f}s audio={dur:.2f}s bytes={n}")
w, fa, dur, n = req(PARA, True, f"{OUT}/{LABEL}_para1.pcm")
log(f"para1 stream wall={w:.2f}s ttfa={fa:.2f}s audio={dur:.2f}s rtf={w/dur:.2f} bytes={n}")
log("DONE " + LABEL)
RES.close()
