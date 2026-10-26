from pathlib import Path
import importlib.util
import pytest


def test_public_boundary_accepts_and_rejects():
    path = Path(__file__).resolve().parents[1] / "tools/check_public.py"
    spec = importlib.util.spec_from_file_location("boundary", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.check_paths(["src/jointshrink/model.py", "docs/manual/jointshrink-reference.pdf"])
    for bad in ("private/plan.md", "docs/author.tex", "/tmp/data.md", "docs/../private/a.md"):
        with pytest.raises(ValueError):
            mod.check_paths([bad])


def boundary_module():
    path = Path(__file__).resolve().parents[1] / "tools/check_public.py"
    spec = importlib.util.spec_from_file_location("boundary_pdf", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("kind", ["link", "metadata", "relative_link"])
def test_pdf_boundary_rejects_decoded_local_paths(tmp_path, kind):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    local = "/" + "Users/example/private/data.csv"
    if kind == "link":
        writer.add_uri(0, "file" + "://" + local, [0, 0, 10, 10])
    elif kind == "relative_link":
        writer.add_uri(0, "../reference.md", [0, 0, 10, 10])
    else:
        writer.add_metadata({"/Subject": local})
    writer.compress_identical_objects()
    path = tmp_path / "local.pdf"
    writer.write(path)
    with pytest.raises(ValueError, match="Local path|Non-public PDF link"):
        boundary_module().check_pdf(path)


def test_pdf_boundary_accepts_web_links_and_metadata(tmp_path):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_uri(0, "https://example.org/data.csv", [0, 0, 10, 10])
    writer.add_metadata({"/Author": "Researcher", "/Title": "Reference manual"})
    path = tmp_path / "web.pdf"
    writer.write(path)
    boundary_module().check_pdf(path)


def test_manual_assets_and_links_have_independent_public_destinations():
    path = Path(__file__).resolve().parents[1] / "tools/build_manual.py"
    spec = importlib.util.spec_from_file_location("manual_builder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tex = r"\includegraphics{../figures/curve.png} \href{../figures/data.csv}{Data} \href{../reference-checks.md}{Checks} \href{https://example.org/}{Source}"
    expected = r"\includegraphics{figures/curve.png} \href{https://nilson01.github.io/jointshrink/figures/data.csv}{Data} \href{https://nilson01.github.io/jointshrink/reference-checks/}{Checks} \href{https://example.org/}{Source}"
    assert mod.prepare_tex(tex) == expected
