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


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_distribution_boundary_checks_members_and_content(tmp_path, kind):
    import io
    import tarfile
    import zipfile

    boundary = boundary_module()
    for member, content, valid in [
        ("src/jointshrink/model.py", b"value = 1", True),
        ("private/notes.md", b"private", False),
        ("src/jointshrink/model.py", b'path = "/' + b'Users/example/data"', False),
        ("../escape.py", b"value = 1", False),
    ]:
        if kind == "wheel":
            path = tmp_path / "jointshrink-0.1.0-py3-none-any.whl"
            name = member.removeprefix("src/")
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(name, content)
        else:
            path = tmp_path / "jointshrink-0.1.0.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                item = tarfile.TarInfo("jointshrink-0.1.0/" + member)
                item.size = len(content)
                archive.addfile(item, io.BytesIO(content))
        if valid:
            assert boundary.check_archive(path) == 1
        else:
            with pytest.raises(ValueError):
                boundary.check_archive(path)


def test_distribution_boundary_rejects_pdf_annotation(tmp_path):
    import io
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_uri(0, "file" + ":///" + "Users/example/private.csv", [0, 0, 10, 10])
    pdf = io.BytesIO()
    writer.write(pdf)
    path = tmp_path / "jointshrink-0.1.0.tar.gz"
    import tarfile
    with tarfile.open(path, "w:gz") as archive:
        item = tarfile.TarInfo("jointshrink-0.1.0/docs/manual/reference.pdf")
        item.size = len(pdf.getvalue())
        archive.addfile(item, io.BytesIO(pdf.getvalue()))
    with pytest.raises(ValueError, match="Non-public PDF link"):
        boundary_module().check_archive(path)


@pytest.mark.parametrize("name", [
    "jointshrink/./model.py",
    "jointshrink/private/notes.py",
    "jointshrink-0.1.0.dist-info/personal-plan.md",
])
def test_wheel_boundary_rejects_aliases_and_unlisted_members(tmp_path, name):
    import zipfile
    path = tmp_path / "jointshrink-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("jointshrink/model.py", "value = 1")
        archive.writestr(name, "value = 1")
    with pytest.raises(ValueError):
        boundary_module().check_archive(path)


@pytest.mark.parametrize("names,valid", [
    (["jointshrink-0.1.0-py3-none-any.whl", "jointshrink-0.1.0.tar.gz"], True),
    (["jointshrink-0.1.0-py3-none-any.whl", "jointshrink-0.2.0-py3-none-any.whl"], False),
    (["jointshrink-0.1.0-py3-none-any.whl", "jointshrink-0.2.0.tar.gz"], False),
    (["jointshrink-0.1.0-py3-none-any.whl", "jointshrink-0.1.0.tar.gz", "notes.md"], False),
])
def test_release_distribution_pair_identity(tmp_path, names, valid):
    for name in names:
        (tmp_path / name).touch()
    if valid:
        assert len(boundary_module().distribution_paths(tmp_path)) == 2
    else:
        with pytest.raises(ValueError):
            boundary_module().distribution_paths(tmp_path)
