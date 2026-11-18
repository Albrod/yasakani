# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""``yasakani update [--check]`` (0.1.7): is there a newer release, and how to get it.

This is the only part of the package that contacts the network, and only when the command is typed.  It sends one HTTPS GET to
GitHub's API for the latest release, with a generic User-Agent and nothing about this computer.  ``serve``, ``chat`` and ``bench``
never call it.

Inside a folder made by the quickstart (START.cmd) the update itself is the launcher's job: this command opens a new window with
START.cmd --action update and ends, so that nothing of this environment is in use when the launcher (which runs on the system
Python, not on this environment) replaces the package.
Otherwise it prints what to download and the command to run, and changes nothing.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from . import __version__

API = "https://api.github.com/repos/yasakani-dev/yasakani/releases/latest"
REPO = "https://github.com/yasakani-dev/yasakani"
USER_AGENT = "yasakani-update"
MAX_ANSWER = 1 << 20


def parse_version(s) -> Optional[tuple]:
    """(0, 1, 10) from "0.1.10" or "v0.1.10"; None for anything else (pre-release tags included)."""
    m = re.fullmatch(r"v?(\d{1,4})\.(\d{1,4})\.(\d{1,4})", str(s or "").strip())
    return tuple(int(x) for x in m.groups()) if m else None


def is_newer(new, old) -> bool:
    a, b = parse_version(new), parse_version(old)
    return a is not None and b is not None and a > b


def latest(api: str = API, timeout: float = 5.0) -> dict:
    """{"version", "tag", "page", "assets": {name: url}} of the latest release.  Raises OSError / ValueError on any problem;
    drafts and pre-releases are refused."""
    req = urllib.request.Request(api, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read(MAX_ANSWER + 1)
    if len(data) > MAX_ANSWER:
        raise ValueError("the answer of the release API is too large")
    j = json.loads(data.decode("utf-8"))
    if not isinstance(j, dict) or j.get("draft") or j.get("prerelease"):
        raise ValueError("the latest release is not a published release")
    tag = j.get("tag_name")
    v = parse_version(tag)
    if not (isinstance(tag, str) and tag.startswith("v") and v):
        raise ValueError("the latest release has an unexpected tag")
    assets = {x["name"]: x["browser_download_url"] for x in j.get("assets") or []
              if isinstance(x, dict) and isinstance(x.get("name"), str) and isinstance(x.get("browser_download_url"), str)}
    return {"version": "%d.%d.%d" % v, "tag": tag, "page": REPO + "/releases/tag/" + tag, "assets": assets}


def quickstart_install(prefix: Optional[str] = None) -> Optional[dict]:
    """When this Python environment is the ``venv`` of a quickstart folder: {"home", "launcher", "python"}; else None.  ``python``
    is the Python the environment was made from (pyvenv.cfg), which is the one START.cmd runs."""
    env = Path(prefix or sys.prefix)
    home = env.parent
    if env.name != "venv" or not (home / "quickstart_state.json").is_file():
        return None
    try:
        st = json.loads((home / "quickstart_state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = {}
    launcher = None
    for d in (st.get("quickstart_dir") if isinstance(st, dict) else None, str(home)):
        if d and (Path(d) / "quickstart.py").is_file():
            launcher = Path(d) / "quickstart.py"
            break
    if launcher is None:
        return None
    python = None
    try:
        for line in (env / "pyvenv.cfg").read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip().lower() == "home" and v.strip():
                for exe in ("python.exe", "python3", "python"):
                    if (Path(v.strip()) / exe).is_file():
                        python = str(Path(v.strip()) / exe)
                        break
    except OSError:
        pass
    python = python or getattr(sys, "_base_executable", None) or sys.executable
    return {"home": str(home), "launcher": str(launcher), "python": python}


def manual_steps(rel: dict) -> List[str]:
    """What a manual install does to update: two steps."""
    if sys.platform == "win32":
        z = "yasakani-%s-win-x64.zip" % rel["version"]
        url = (rel.get("assets") or {}).get(z) or "%s/releases/download/%s/%s" % (REPO, rel["tag"], z)
        return ["1. download and unpack %s" % url,
                "   (its SHA-256 is in %s/releases/download/%s/SHA256SUMS.txt)" % (REPO, rel["tag"]),
                "2. pip install --upgrade <the unpacked folder>   and use the yskn.dll that is in it"]
    return ["1. git pull   (or download the source of %s)" % rel["tag"],
            "2. build the engine and pip install --upgrade ./python again, as in BUILDING.md"]


def cmd_update(args, api: str = API, prefix: Optional[str] = None, run: Callable[[Sequence[str]], int] = subprocess.call,
               out=None) -> int:
    """0: up to date, updated, or the steps were printed; 1: the check failed or the launcher reported an error."""
    out = out or sys.stdout
    qs = None if args.check else quickstart_install(prefix)
    if qs is not None:
        start = Path(qs["launcher"]).with_name("START.cmd")
        if sys.platform == "win32" and start.is_file():
            # a new window runs START.cmd --action update, and this command ends at once: while it ran, Windows would keep
            # this environment's python.exe and yasakani.exe in use, and the update reinstalls the package of this environment
            cmd = ["cmd.exe", "/c", "start", "YASAKANI update", str(start), "--action", "update", "--home", qs["home"]]
            cmd += ["--yes"] if args.yes else []
            print("This folder was set up by the quickstart.  Its launcher does the update in a new window:", file=out)
            print("  %s --action update" % start, file=out, flush=True)
            return 1 if run(cmd) else 0
        cmd = [qs["python"], qs["launcher"], "--action", "update", "--home", qs["home"]] + (["--yes"] if args.yes else [])
        print("This folder was set up by the quickstart.  Its launcher does the update:", file=out)
        print("  %s %s --action update" % (qs["python"], qs["launcher"]), file=out, flush=True)
        return 1 if run(cmd) else 0
    print("Asking github.com for the latest release (one request; nothing about this computer is sent)", file=out, flush=True)
    try:
        rel = latest(api)
    except Exception as ex:  # noqa: BLE001 - offline, rate limit, ...
        print("could not check: %s" % type(ex).__name__, file=out)
        return 1
    print("installed  %s" % __version__, file=out)
    print("latest     %s   %s" % (rel["version"], rel["page"]), file=out)
    if not is_newer(rel["version"], __version__):
        print("This is the latest version.", file=out)
        return 0
    if args.check:
        return 0
    print("To update:", file=out)
    for line in manual_steps(rel):
        print("  " + line, file=out)
    print("Nothing was changed.", file=out)
    return 0
