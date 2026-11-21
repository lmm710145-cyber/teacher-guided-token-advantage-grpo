"""Fetch the exact compiled PDDL assets used by this protocol, with checksums."""

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "json_2.1.1_json.zip": "25171f16e20ad7b048c47275c45b0babf3aa1cbab29cec97387922350a9844bc",
    "json_2.1.1_pddl.zip": "913942ebed06659ea0da2f8122512d98bc6add30d84961ca803132d8fbcad585",
    "json_2.1.1_tw-pddl.zip": "d376c4bbb097c068528a2a9805e358e4cd14bc75c4d68a24e5141aebd234ef67",
}


def main():
    cache = ROOT / "cache/downloads"
    cache.mkdir(parents=True, exist_ok=True)
    dest = ROOT / "data/environments/alfworld-0.3.3"
    dest.mkdir(parents=True, exist_ok=True)
    for name, sha in ASSETS.items():
        file = cache / name
        if not file.exists():
            temp = file.with_suffix(".partial")
            urllib.request.urlretrieve(
                "https://github.com/alfworld/alfworld/releases/download/0.2.2/" + name,
                temp,
            )
            temp.rename(file)
        assert hashlib.sha256(file.read_bytes()).hexdigest() == sha, (
            f"Asset checksum mismatch: {name}"
        )
        with zipfile.ZipFile(file) as z:
            for member in z.infolist():
                target = (dest / member.filename).resolve()
                if not target.is_relative_to(dest.resolve()):
                    raise ValueError("Unsafe archive path")
            z.extractall(dest)
    logic = dest / "logic"
    logic.mkdir(exist_ok=True)
    for name in ("alfred.pddl", "alfred.twl2"):
        shutil.copyfile(ROOT / "benchmarks/alfworld/alfworld/data" / name, logic / name)


if __name__ == "__main__":
    main()
