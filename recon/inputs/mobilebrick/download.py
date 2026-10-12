#!/usr/bin/env python3
"""Stage every MobileBrick held-out scene from the official ZIP using HTTP ranges.

Dataset: Li et al., "MobileBrick: Building LEGO for 3D Reconstruction on Mobile
Devices", CVPR 2023 (https://code.active.vision/MobileBrick/); use is subject
to the dataset's own terms.
"""
from __future__ import annotations

import argparse
import binascii
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import struct
import time
import zlib

import requests
from remotezip import RemoteZip
from recon.common.progress import bar, done

URL = 'https://www.robots.ox.ac.uk/~victor/data/MobileBrick/MobileBrick_Mar23.zip'
LOCAL_HEADER = struct.Struct('<IHHHHHIIIHH')


def fetch_scene(name: str, entries: list, destination: Path) -> dict:
    start = min(item.header_offset for item in entries)
    end = max(item.header_offset + item.compress_size + 4096 for item in entries)
    compressed_dir = destination / '_compressed'
    compressed_dir.mkdir(parents=True, exist_ok=True)
    span = compressed_dir / f'{name}.zipspan'
    expected = end - start + 1
    for attempt in range(5):
        current = span.stat().st_size if span.exists() else 0
        if current == expected:
            break
        if current > expected:
            span.unlink()
            current = 0
        response = requests.get(URL, headers={'Range': f'bytes={start+current}-{end}'},
                                timeout=120, stream=True)
        try:
            response.raise_for_status()
            content_range = response.headers.get('Content-Range', '')
            if response.status_code != 206 or not content_range.startswith(f'bytes {start+current}-{end}/'):
                raise IOError(f'Unexpected HTTP range response: {content_range}')
            with span.open('ab') as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
        finally:
            response.close()
        if span.stat().st_size == expected:
            break
        time.sleep(1 + attempt)
    if span.stat().st_size != expected:
        raise IOError(f'Incomplete download for {name}: {span.stat().st_size}/{expected}')

    extracted = 0
    with span.open('rb') as stream:
        for item in entries:
            if item.is_dir():
                continue
            target = destination / item.filename
            if target.is_file() and target.stat().st_size == item.file_size:
                continue
            stream.seek(item.header_offset - start)
            header = stream.read(LOCAL_HEADER.size)
            values = LOCAL_HEADER.unpack(header)
            if values[0] != 0x04034b50:
                raise ValueError(f'Invalid ZIP local header: {item.filename}')
            name_length, extra_length = values[-2:]
            stream.seek(name_length + extra_length, 1)
            payload = stream.read(item.compress_size)
            if len(payload) != item.compress_size:
                raise IOError(f'Truncated ZIP entry: {item.filename}')
            if item.compress_type == 8:
                content = zlib.decompress(payload, -15)
            elif item.compress_type == 0:
                content = payload
            else:
                raise ValueError(f'Unsupported ZIP method {item.compress_type}')
            if len(content) != item.file_size or binascii.crc32(content) != item.CRC:
                raise IOError(f'ZIP checksum mismatch: {item.filename}')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.part')
            temporary.write_bytes(content)
            temporary.replace(target)
            extracted += 1
    span.unlink()
    return {'scene': name, 'files': len(entries), 'extracted': extracted,
            'download_mb': round(expected / 1e6, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('runs/real/mobilebrick_dataset'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    with RemoteZip(URL) as archive:
        groups = defaultdict(list)
        for item in archive.infolist():
            parts = item.filename.split('/')
            if len(parts) >= 3 and parts[0] == 'test':
                groups[parts[1]].append(item)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_scene, name, entries, args.output): name
                   for name, entries in groups.items()}
        for future in bar(as_completed(futures), total=len(futures),
                          desc='Downloading MobileBrick scenes', unit='scene'):
            future.result()
    done(f'Downloaded {len(groups)} MobileBrick test scenes to {args.output}')


if __name__ == '__main__':
    main()
