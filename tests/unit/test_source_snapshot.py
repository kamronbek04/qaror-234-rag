import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_committed_snapshot_matches_recorded_checksum():
    snapshot = ROOT / "data" / "raw" / "lex_8193120.html"
    source_note = (ROOT / "data" / "raw" / "SOURCE.md").read_text(encoding="utf-8")

    recorded = re.search(r"SHA-256:\s*`([0-9a-f]{64})`", source_note)

    assert recorded, "SOURCE.md must record the snapshot SHA-256"
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == recorded.group(1)
