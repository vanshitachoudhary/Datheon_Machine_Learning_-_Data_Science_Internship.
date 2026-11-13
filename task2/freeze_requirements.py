"""Pins every package in requirements.txt to the exact version installed on THIS computer.
Why: the saved model (.joblib) must be loaded by the same scikit-learn/pandas/numpy versions that trained it.
Run it after the notebook, before pushing to GitHub:   python freeze_requirements.py"""
import re
import importlib.metadata as md

def main(path="requirements.txt"):
    out, missing = [], []
    for line in open(path, encoding="utf-8").read().splitlines():
        name = re.split(r"[=<>~!\s]", line.strip(), maxsplit=1)[0]
        if not name or name.startswith("#") or name.startswith("-"):
            out.append(line); continue
        try:
            out.append(f"{name}=={md.version(name)}")
        except md.PackageNotFoundError:
            out.append(name); missing.append(name)
    open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("requirements.txt pinned to your installed versions.")
    if missing:
        print("Not installed here (left unpinned):", ", ".join(missing))

if __name__ == "__main__":
    main()
