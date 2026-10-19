"""Quick structural checks for the synthetic memory dataset."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
bio = json.loads((ROOT / "data" / "synthetic_biography.json").read_text(encoding="utf-8"))
heldout = json.loads((ROOT / "data" / "heldout_questions.json").read_text(encoding="utf-8"))
train = [json.loads(line) for line in (ROOT / "data" / "synthetic_memory_train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

assert bio["person"]["name"] == "Leela Menon"
assert len(bio["timeline"]) == 18
assert len(train) == 83, f"Expected 83 conversations, found {len(train)}"
assert len(heldout) == 42, f"Expected 42 held-out prompts, found {len(heldout)}"
print(f"OK: {len(train)} train conversations, {len(heldout)} held-out prompts, {len(bio['timeline'])} timeline events.")
