"""Check source paths against the reviewed distribution boundary."""

from pathlib import Path, PurePosixPath
import argparse
import io
import stat
import subprocess
import tarfile
import zipfile

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
WHEEL_MODULES = {
    "__init__.py", "_numeric.py", "_sampling.py", "data.py", "io.py",
    "likelihoods.py", "model.py", "patterns.py", "plotting.py", "regression.py",
    "regression_plots.py", "results.py", "student.py",
}
WHEEL_METADATA = {"METADATA", "WHEEL", "top_level.txt", "RECORD", "licenses/LICENSE"}
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



def check_archive(path):
    """Inspect archive members in memory without extracting their paths."""
    path = Path(path)
    wheel = path.name.endswith(".whl")
    seen = set()
    if wheel:
        with zipfile.ZipFile(path) as archive:
            members = []
            for item in archive.infolist():
                if item.is_dir():
                    continue
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("Archive symlink: " + item.filename)
                members.append((item.filename, archive.read(item)))
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            members = []
            for item in archive.getmembers():
                if item.isdir():
                    continue
                if not item.isfile():
                    raise ValueError("Non-regular archive member: " + item.name)
                members.append((item.name, archive.extractfile(item).read()))
    else:
        raise ValueError("Unexpected distribution format: " + path.name)
    for name, content in members:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or str(relative) != name or name in seen:
            raise ValueError("Unsafe or duplicate archive path: " + name)
        seen.add(name)
        if relative.suffix.lower() in {".tex", ".aux", ".log", ".bbl", ".bib", ".zip", ".pem", ".key"}:
            raise ValueError("Unexpected distribution file: " + name)
        if wheel:
            expected_info = path.name.split("-", 2)
            info = expected_info[0] + "-" + expected_info[1] + ".dist-info"
            if len(relative.parts) == 2 and relative.parts[0] == "jointshrink" and relative.parts[1] in WHEEL_MODULES:
                pass
            elif relative.parts[0] == info and str(PurePosixPath(*relative.parts[1:])) in WHEEL_METADATA:
                pass
            else:
                raise ValueError("Unexpected wheel member: " + name)
        else:
            expected_root = path.name[:-7]
            if len(relative.parts) < 2 or relative.parts[0] != expected_root:
                raise ValueError("Unexpected source archive root: " + name)
            relative = PurePosixPath(*relative.parts[1:])
            if str(relative) not in {"PKG-INFO", "setup.cfg"}:
                check_paths([str(relative)])
        if relative.suffix.lower() == ".pdf":
            check_pdf(io.BytesIO(content))
        elif relative.suffix in {".py", ".md", ".yml", ".toml", ".txt", ".html", ".R", ".cff", ".json"} or relative.name in {"LICENSE", "METADATA", "PKG-INFO", "WHEEL"}:
            check_text(content.decode("utf-8"), name)
    if not members:
        raise ValueError("Empty distribution archive")
    return len(members)


def distribution_paths(directory):
    """Require one matching pure-Python wheel/source pair and no other files."""
    files = sorted(Path(directory).iterdir())
    wheels = [p for p in files if p.name.endswith(".whl")]
    sources = [p for p in files if p.name.endswith(".tar.gz")]
    if len(files) != 2 or len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Expected exactly one wheel and one source archive")
    stem = sources[0].name[:-7]
    if not stem.startswith("jointshrink-") or wheels[0].name != stem + "-py3-none-any.whl":
        raise ValueError("Distribution names or versions do not match")
    return wheels + sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path)
    args = parser.parse_args()
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
    if args.archives is not None:
        for archive in distribution_paths(args.archives):
            print(f"{archive.name}: {check_archive(archive)} reviewed members")
    print("Source paths and text markers satisfy the reviewed distribution boundary.")


if __name__ == "__main__":
    main()
