"""Check source paths against the reviewed distribution boundary."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    ".gitignore",
    ".gitattributes",
    "README.md",
    "LICENSE",
    "NOTICE.md",
    "CITATION.cff",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "MANIFEST.in",
    "pyproject.toml",
    "mkdocs.yml",
}
PREFIXES = (".github/", "src/", "tests/", "tools/", "examples/", "docs/", "overrides/")


def check_paths(paths):
    errors = []
    for name in paths:
        p = Path(name)
        if (
            p.is_absolute()
            or ".." in p.parts
            or not (name in ROOT_FILES or name.startswith(PREFIXES))
        ):
            errors.append(name)
        elif p.suffix.lower() in {".tex", ".aux", ".log", ".bbl", ".bib", ".zip", ".pem", ".key"}:
            errors.append(name)
    if errors:
        raise ValueError("Unexpected source paths: " + ", ".join(errors))


def check_text(text, name):
    markers = ("/" + "Users/", "/" + "var/folders/", "file" + ":/", "BEGIN " + "PRIVATE KEY")
    if any(marker in text for marker in markers):
        raise ValueError("Local path or credential marker in " + str(name))


def check_pdf(path):
    """Inspect decoded PDF objects, including compressed link annotations."""
    from pypdf import PdfReader
    from pypdf.generic import IndirectObject

    reader = PdfReader(path, strict=True)
    seen = set()

    def inspect(value):
        if isinstance(value, IndirectObject):
            key = (value.idnum, value.generation)
            if key in seen:
                return
            seen.add(key)
            inspect(value.get_object())
        elif isinstance(value, dict):
            if "/URI" in value:
                uri = str(value["/URI"])
                if not uri.startswith(("https://", "http://", "mailto:")):
                    raise ValueError("Non-public PDF link in " + str(path))
            for item in value.values():
                inspect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                inspect(item)
        elif isinstance(value, str):
            check_text(value, path)

    inspect(reader.trailer)


def main():
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True
    ).splitlines()
    check_paths(paths)
    for name in paths:
        path = ROOT / name
        if path.is_symlink():
            raise ValueError("Symlinks require explicit distribution review: " + name)
        if path.suffix.lower() == ".pdf":
            check_pdf(path)
        elif path.suffix in {".py", ".md", ".yml", ".toml", ".txt", ".html", ".R", ".cff", ".json"}:
            check_text(path.read_text(encoding="utf-8"), name)
    print("Source paths and text markers satisfy the reviewed distribution boundary.")


if __name__ == "__main__":
    main()
