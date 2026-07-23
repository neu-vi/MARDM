from pathlib import Path

TEXTS_DIR = Path("texts")
OUT_DIR = Path("descriptions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for in_path in sorted(TEXTS_DIR.glob("*.txt")):
    descriptions = []

    for line in in_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue

        # Each line is one annotation: description#processed#start#end
        parts = [p.strip() for p in line.split("#")]

        # Be tolerant: take the first field as description if present
        if len(parts) >= 1 and parts[0]:
            descriptions.append(parts[0])

    out_path = OUT_DIR / in_path.name
    out_path.write_text(
        "\n".join(descriptions) + ("\n" if descriptions else ""),
        encoding="utf-8",
    )
