"""Stream-download the S2-SHIPS dataset asset with progress, bypassing eotdl's
non-streaming stage_file_url (which buffers the whole file in memory with no
progress output). Verifies the SHA1 checksum from the STAC catalog afterward.
"""
import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import requests
from tqdm import tqdm

from eotdl.repos import FilesAPIRepo
from eotdl.auth import auth


def get_presigned_url(dataset_id: str, file_name: str, user) -> str:
    repo = FilesAPIRepo()
    url = repo.url + f"datasets/{dataset_id}/stage/{file_name}"
    response = requests.get(url, headers=repo.generate_headers(user))
    data, error = repo.format_response(response)
    if error:
        raise Exception(error)
    return data["presigned_url"]


def stream_download(url: str, dest: Path, expected_size: int | None):
    tmp = dest.with_suffix(dest.suffix + ".part")
    resume_from = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if resume_from and r.status_code == 206:
            mode = "ab"
            total = expected_size
        else:
            mode = "wb"
            resume_from = 0
            total = int(r.headers.get("content-length", expected_size or 0))
        r.raise_for_status()
        with open(tmp, mode) as f, tqdm(
            total=total, initial=resume_from, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    tmp.rename(dest)


def verify_checksum(path: Path, expected_sha1: str) -> bool:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    actual = h.hexdigest()
    print(f"checksum: expected={expected_sha1} actual={actual}")
    return actual == expected_sha1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/S2-SHIPS")
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    catalog_path = data_dir / "catalog.v1.parquet"
    if not catalog_path.exists():
        sys.exit(f"catalog not found at {catalog_path} — run `eotdl datasets get S2-SHIPS -p data` first")

    gdf = gpd.read_parquet(catalog_path)
    row = gdf.iloc[0]
    asset = row["assets"]["asset"]
    href = asset["href"]
    size = asset.get("size")
    checksum = asset.get("checksum")

    # href like https://api.eotdl.com/datasets/<id>/stage/<file_name>
    dataset_id = href.split("/datasets/")[1].split("/stage/")[0]
    file_name = href.split("/stage/")[-1]

    user = auth()
    presigned = get_presigned_url(dataset_id, file_name, user)

    dest = data_dir / file_name
    print(f"downloading {file_name} ({size/1e9:.2f} GB) -> {dest}")
    stream_download(presigned, dest, size)

    if checksum:
        ok = verify_checksum(dest, checksum)
        if not ok:
            sys.exit("checksum mismatch — download is corrupt, re-run to retry")

    if not args.skip_extract:
        print(f"extracting {dest} ...")
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(data_dir)
        print("extraction complete")


if __name__ == "__main__":
    main()
