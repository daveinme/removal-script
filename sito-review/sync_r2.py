#!/usr/bin/env python3
"""
Carica su Cloudflare R2 tutte le immagini presenti in output/iterazioni/,
preservando la struttura vN_tag/nome_file.png -> chiave S3 identica.
Idempotente: salta i file già presenti su R2 con stessa dimensione.
"""
import io
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
THUMB_MAX_SIDE = 500  # lato lungo in px, per la griglia di anteprime
THUMB_QUALITY = 70


def _make_thumb_bytes(path: Path) -> bytes:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((THUMB_MAX_SIDE, THUMB_MAX_SIDE), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=THUMB_QUALITY)
        return buf.getvalue()


def get_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def sync():
    bucket = os.environ["R2_BUCKET_NAME"]
    public_url = os.environ["R2_PUBLIC_URL"].rstrip("/")
    client = get_client()

    iter_dir = ROOT / "output" / "iterazioni"
    if not iter_dir.exists():
        print("Nessuna cartella iterazioni trovata.")
        return []

    existing = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="iterazioni/"):
        for obj in page.get("Contents", []):
            existing[obj["Key"]] = obj["Size"]

    uploaded = []
    for path in sorted(iter_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = str(path.relative_to(iter_dir)).replace(os.sep, "/")
        key = "iterazioni/" + rel
        size = path.stat().st_size
        if existing.get(key) != size:
            content_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": content_type})
            uploaded.append(key)
            print(f"caricato: {key}")

        thumb_key = "thumbs/" + str(Path(rel).with_suffix(".jpg"))
        if thumb_key not in existing:
            thumb_bytes = _make_thumb_bytes(path)
            client.put_object(
                Bucket=bucket, Key=thumb_key, Body=thumb_bytes, ContentType="image/jpeg",
            )
            uploaded.append(thumb_key)
            print(f"caricata thumb: {thumb_key}")

    if not uploaded:
        print("Nessun file nuovo da caricare, tutto già sincronizzato.")
    else:
        print(f"Caricati {len(uploaded)} file su R2 (bucket: {bucket}).")
    return uploaded


if __name__ == "__main__":
    sync()
