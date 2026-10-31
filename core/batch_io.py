"""Batch input: turns uploaded files into a list of students.

  * a normal file          -> one student (name = file name)
  * a .zip                 -> each top-level file is one student, and each
                              top-level folder is one student whose pages
                              are the files inside, sorted by name
Zip files are extracted safely: no paths outside the target folder, no
links, and limits on file count and size."""

import zipfile
from pathlib import Path

from .config import load_config

SKIP = {".ds_store", "thumbs.db", "desktop.ini"}


def _safe_extract(zip_path: Path, dest: Path) -> None:
    """Extract a zip into a folder, skipping links and paths that leave the folder.

    Args:
        zip_path (Path): The uploaded zip.
        dest (Path): Folder to extract into.

    Raises:
        ValueError: If the zip has more files or more unpacked bytes than the
            batch limits in config.yaml (ingest.batch_max_files, ingest.batch_max_mb).
    """
    cfg = load_config()["ingest"]
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        if len(members) > cfg["batch_max_files"]:
            raise ValueError(f"Zip has more than {cfg['batch_max_files']} files.")
        if sum(m.file_size for m in members) > cfg["batch_max_mb"] * 1024 * 1024:
            raise ValueError(f"Zip is larger than {cfg['batch_max_mb']} MB when unpacked.")
        for m in members:
            if (m.external_attr >> 16) & 0o170000 == 0o120000:  # symbolic link
                continue
            target = (dest / m.filename).resolve()
            if not str(target).startswith(str(dest)):
                continue  # "../" tricks
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(target, "wb") as out:
                out.write(src.read())


def _usable(p: Path) -> bool:
    """Check that a path is a real file and not system clutter (.DS_Store, __MACOSX, ...).

    Args:
        p (Path): Path inside the extracted zip.

    Returns:
        bool: True if the file should be graded.
    """
    return p.is_file() and p.name.lower() not in SKIP and not p.name.startswith((".", "__MACOSX"))


def collect_students(paths: list[str], workdir: str | Path) -> list[tuple[str, list[str]]]:
    """Turn uploaded files and zips into a list of students and their pages.

    Args:
        paths (list[str]): Uploaded files.
        workdir (str | Path): Folder where zips are extracted.

    Returns:
        list[tuple[str, list[str]]]: (student name, page file paths) per student.

    Raises:
        ValueError: If a zip is over the batch limits.
    """
    workdir = Path(workdir)
    students: list[tuple[str, list[str]]] = []
    for raw in paths:
        p = Path(raw)
        if p.suffix.lower() != ".zip":
            students.append((p.stem, [str(p)]))
            continue
        dest = workdir / p.stem
        dest.mkdir(parents=True, exist_ok=True)
        _safe_extract(p, dest)
        root = dest
        entries = [e for e in sorted(root.iterdir()) if e.name != "__MACOSX"]
        if len(entries) == 1 and entries[0].is_dir():  # zip of a single wrapper folder
            root = entries[0]
            entries = sorted(root.iterdir())
        for e in entries:
            if e.is_dir():
                pages = sorted(str(f) for f in e.rglob("*") if _usable(f))
                if pages:
                    students.append((e.name, pages))
            elif _usable(e):
                students.append((e.stem, [str(e)]))
    return students
