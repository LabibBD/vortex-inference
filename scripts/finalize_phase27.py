#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
required = [
    ROOT / "README.md",
    ROOT / "CHANGELOG.md",
    ROOT / "docs" / "architecture.md",
    ROOT / "docs" / "interview_guide.md",
    ROOT / "docs" / "resume_bullets.md",
    ROOT / "docs" / "release_checklist.md",
]

missing = [path for path in required if not path.exists()]
if missing:
    raise SystemExit("Missing final files: " + ", ".join(str(p) for p in missing))

print("[VORTEX] Final documentation is present")
for path in required:
    print(f"  {path.relative_to(ROOT)}")
