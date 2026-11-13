"""Run a method on every set, time it, and (optionally) score it.

The command is a template; these placeholders are filled in for each set:
  {images}  folder with the 24 input photos          (sets/t01/public/images)
  {public}  the set's public folder                  (sets/t01/public)
  {out}     where the method must write its 2048x1024 PNG
Time = wall clock from launching the command until it exits (model loading included).
A set that runs longer than --limit seconds is stopped and counts as a failure (score 0).
After every set, all processes the command started are killed -- including ones that detached
into the background -- so nothing (e.g. a pre-loaded model server) survives into the next set.
Run it from a dedicated account/terminal: any other process of the same user started while a
set is running would be killed too.

Example
  python run_eval.py --sets sets --out runs/myteam \
      --cmd "python my_method.py --images {images} --out {out}" --score
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--sets", required=True, help="folder that contains the set folders (t01, t02, ...)")
ap.add_argument("--cmd", required=True, help="command template, see above")
ap.add_argument("--out", required=True, help="output folder for PNGs and summary.json")
ap.add_argument("--limit", type=float, default=300.0, help="time limit per set in seconds")
ap.add_argument("--score", action="store_true", help="also run score.py (needs the hidden/ folder)")
ap.add_argument("--only", nargs="*", default=None, help="run only these set ids")
a = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))


def user_pids():
    out = subprocess.run(["ps", "-U", str(os.getuid()), "-o", "pid="], capture_output=True, text=True).stdout
    return {int(x) for x in out.split()}


def kill_new(before):
    """Kill every process of this user that did not exist before the set started."""
    me = os.getpid()
    for _ in range(3):  # repeat: a dying parent may have spawned more children
        new = user_pids() - before - {me}
        alive = []
        for pid in new:
            try:
                os.kill(pid, signal.SIGKILL)
                alive.append(pid)
            except (ProcessLookupError, PermissionError):
                pass
        if not alive:
            break
        time.sleep(0.2)
    return new

os.makedirs(a.out, exist_ok=True)
ids = sorted(d for d in os.listdir(a.sets) if os.path.isdir(os.path.join(a.sets, d, "public", "images")))
if a.only:
    ids = [i for i in ids if i in a.only]
if not ids:
    sys.exit(f"no sets with public/images found in {a.sets}")

rows = []
for sid in ids:
    public = os.path.join(a.sets, sid, "public")
    out_png = os.path.abspath(os.path.join(a.out, f"{sid}.png"))
    if os.path.exists(out_png):
        os.remove(out_png)
    cmd = a.cmd.format(images=os.path.join(public, "images"), public=public, out=out_png)
    log = open(os.path.join(a.out, f"{sid}.log"), "w")
    before = user_pids()
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        rc = proc.wait(timeout=a.limit)
        status = "ok" if rc == 0 else f"error({rc})"
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)  # stop the whole process group, not just the shell
        proc.wait()
        status = "timeout"
    sec = time.perf_counter() - t0
    log.close()
    leftover = kill_new(before)  # nothing may keep running between sets
    if status == "ok" and not os.path.exists(out_png):
        status = "no_output"
    row = {"set": sid, "seconds": round(sec, 2), "status": status}
    if leftover:
        row["killed_leftover_processes"] = len(leftover)
    if a.score and status == "ok" and os.path.isdir(os.path.join(a.sets, sid, "hidden")):
        js = os.path.join(a.out, f"{sid}.json")
        subprocess.run([sys.executable, os.path.join(here, "score.py"), "--set", os.path.join(a.sets, sid),
                        "--pred", out_png, "--json", js], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(js):
            r = json.load(open(js))
            row.update({k: r[k] for k in ("psnr", "lpips", "tilt_deg", "score")})
    if status != "ok":
        row["score"] = 0.0
    rows.append(row)
    print(f"{sid}  {status:10s}  {sec:7.1f} s" + (f"  quality {row['score']:.1f}" if "score" in row else ""), flush=True)

ok_t = [r["seconds"] if r["status"] == "ok" else a.limit for r in rows]
summary = {"cmd": a.cmd, "limit_s": a.limit, "mean_seconds": round(sum(ok_t) / len(ok_t), 2), "sets": rows}
if all("score" in r for r in rows):
    summary["mean_quality"] = round(sum(r["score"] for r in rows) / len(rows), 2)
json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2, ensure_ascii=False)
print(f"mean time {summary['mean_seconds']} s" + (f", mean quality {summary['mean_quality']}" if "mean_quality" in summary else ""))
