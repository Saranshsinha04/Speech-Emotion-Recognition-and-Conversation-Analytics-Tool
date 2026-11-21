"""Reproducible dataset + build-tool download script (stdlib only).

Downloads:
  * RAVDESS speech audio (Zenodo record 1188976, CC BY-NC-SA 4.0)
  * CREMA-D AudioWAV (GitHub CheyneyComputerScience/CREMA-D, ODbL / DbCL)
  * Apache Maven 3.9.9 binary (build tool, placed in tools/)

Usage:  python research/scripts/download_data.py [--skip-crema] [--skip-ravdess] [--skip-maven]
Files are written under data/raw/ and tools/ (both git-ignored).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
TOOLS = ROOT / "tools"

RAVDESS_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
MAVEN_URL = "https://archive.apache.org/dist/maven/maven-3/3.9.9/binaries/apache-maven-3.9.9-bin.zip"
CREMA_TREE = "https://api.github.com/repos/CheyneyComputerScience/CREMA-D/git/trees/master?recursive=1"
CREMA_MEDIA = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/{path}"


def fetch(url: str, dest: Path, retries: int = 4) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ser-research-downloader"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return dest
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt + 1} for {url}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_ravdess() -> None:
    z = fetch(RAVDESS_URL, RAW / "ravdess" / "Audio_Speech_Actors_01-24.zip")
    out = RAW / "ravdess" / "audio"
    if not out.exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
    n = len(list(out.rglob("*.wav")))
    (RAW / "ravdess" / "PROVENANCE.json").write_text(json.dumps({
        "source": RAVDESS_URL, "license": "CC BY-NC-SA 4.0",
        "citation": "Livingstone & Russo (2018) PLoS ONE 13(5): e0196391",
        "zip_sha256": sha256(z), "n_wav": n}, indent=2))
    print(f"RAVDESS: {n} wav files")


def get_crema() -> None:
    out = RAW / "crema_d" / "AudioWAV"
    tree_json = RAW / "crema_d" / "tree.json"
    fetch(CREMA_TREE, tree_json)
    tree = json.loads(tree_json.read_text())
    paths = [t["path"] for t in tree["tree"] if t["path"].startswith("AudioWAV/") and t["path"].endswith(".wav")]
    print(f"CREMA-D: {len(paths)} files listed (tree truncated={tree.get('truncated')})")

    def one(p: str) -> str:
        fetch(CREMA_MEDIA.format(path=p), out / Path(p).name)
        return p

    done = 0
    with cf.ThreadPoolExecutor(max_workers=24) as ex:
        for _ in ex.map(one, paths):
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(paths)}", flush=True)
    n = len(list(out.glob("*.wav")))
    (RAW / "crema_d" / "PROVENANCE.json").write_text(json.dumps({
        "source": "https://github.com/CheyneyComputerScience/CREMA-D",
        "license": "Open Database License (ODbL) v1.0 / Database Contents License (DbCL) v1.0",
        "citation": "Cao et al. (2014) IEEE Trans. Affective Computing 5(4):377-390",
        "n_wav": n}, indent=2))
    print(f"CREMA-D: {n} wav files")


def get_maven() -> None:
    z = fetch(MAVEN_URL, TOOLS / "apache-maven-3.9.9-bin.zip")
    if not (TOOLS / "apache-maven-3.9.9").exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(TOOLS)
    print("Maven ready at tools/apache-maven-3.9.9")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-crema", action="store_true")
    ap.add_argument("--skip-ravdess", action="store_true")
    ap.add_argument("--skip-maven", action="store_true")
    a = ap.parse_args()
    if not a.skip_maven:
        get_maven()
    if not a.skip_ravdess:
        get_ravdess()
    if not a.skip_crema:
        get_crema()
