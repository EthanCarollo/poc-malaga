#!/usr/bin/env python3
"""Remove execution output from the versioned notebooks without running cells."""
import json
from pathlib import Path


def main():
    changed = []
    for path in sorted(Path(__file__).parent.glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        notebook.get("metadata", {}).pop("widgets", None)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
                cell.get("metadata", {}).pop("execution", None)
        content = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
        if path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8", newline="\n")
            changed.append(path.name)
    print(json.dumps({"cleaned": changed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
