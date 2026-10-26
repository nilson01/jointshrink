"""Compile the maintained Markdown reference with Pandoc and pdfLaTeX."""

from pathlib import Path
import argparse
import posixpath
import re
from urllib.parse import urlsplit
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/manual/reference.md"
HEADER = r"""\usepackage{fvextra}
\usepackage{float}
\usepackage{etoolbox}
\BeforeBeginEnvironment{Shaded}{\par\noindent\begin{minipage}{\linewidth}}
\AfterEndEnvironment{Shaded}{\end{minipage}\par\smallskip}
\usepackage{microtype}
\usepackage{bookmark}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\{\},fontsize=\small}
\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue,citecolor=black,bookmarksdepth=2}
\setlength{\emergencystretch}{3em}
"""


def prepare_tex(tex):
    """Keep assets relative and send external manual links to the website."""

    def public_link(match):
        target = match.group(1)
        parsed = urlsplit(target)
        if parsed.scheme or target.startswith("#"):
            return match.group(0)
        path = posixpath.normpath("manual/" + parsed.path)
        if path.startswith("../") or path.startswith("/"):
            raise ValueError("manual link leaves the documentation directory")
        if path.endswith(".md"):
            path = path[:-3] + "/"
        target = "https://nilson01.github.io/jointshrink/" + path
        if parsed.fragment:
            target += "#" + parsed.fragment
        return r"\href{" + target + "}"

    tex = re.sub(r"\\href\{([^}]+)\}", public_link, tex)
    return tex.replace("../figures/", "figures/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=SOURCE.with_name("jointshrink-reference.pdf")
    )
    target = parser.parse_args().output.resolve()
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        import pypandoc

        pandoc = pypandoc.get_pandoc_path()
    latex = shutil.which("pdflatex")
    if latex is None:
        raise SystemExit("Install a TeX distribution providing pdflatex.")
    with tempfile.TemporaryDirectory(prefix="jointshrink-manual-") as directory:
        work = Path(directory)
        (work / "header.tex").write_text(HEADER)
        subprocess.run(
            [
                pandoc,
                str(SOURCE),
                "--from=markdown+tex_math_dollars",
                "--standalone",
                "--toc",
                "--toc-depth=2",
                "--number-sections",
                "-V",
                "documentclass=article",
                "-V",
                "fontsize=10pt",
                "-V",
                "geometry=margin=0.85in",
                "-V",
                "date=",
                "--include-in-header=" + str(work / "header.tex"),
                "-o",
                str(work / "main.tex"),
            ],
            check=True,
        )
        shutil.copytree(ROOT / "docs/figures", work / "figures")
        tex = prepare_tex((work / "main.tex").read_text())
        tex = tex.replace(r"\begin{figure}", r"\begin{figure}[H]")
        (work / "main.tex").write_text(tex)
        for _ in range(3):
            result = subprocess.run(
                [latex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                cwd=work,
                capture_output=True,
                text=True,
            )
            if result.returncode:
                raise SystemExit(result.stdout[-6000:])
        shutil.copyfile(work / "main.pdf", target)
    print(target)


if __name__ == "__main__":
    main()
