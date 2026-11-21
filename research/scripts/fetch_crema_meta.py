"""Download CREMA-D metadata CSVs and license from the official GitHub repository."""
from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CREMA_DIR = ROOT / "data" / "raw" / "crema_d"
BASE_URL = "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/"

META_FILES = [
    "LICENSE.txt",
    "VideoDemographics.csv",
    "SentenceFilenames.csv",
    "processedResults/summaryTable.csv",
    "processedResults/tabulatedVotes.csv",
]


def main() -> None:
    for rel in META_FILES:
        dest = CREMA_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = BASE_URL + rel
        print(f"Downloading {url} -> {dest}")
        req = urllib.request.Request(url, headers={"User-Agent": "ser-research-downloader"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        dest.write_bytes(data)
        print(f"  saved {len(data)} bytes")


if __name__ == "__main__":
    main()
