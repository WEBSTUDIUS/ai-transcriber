"""Scan the index (default), or reachable commit history, without printing secrets."""
from __future__ import annotations

import argparse
from pathlib import PurePosixPath
import re
import subprocess

PATTERNS = [
    re.compile(rb'hf_[A-Za-z0-9]{20,}'),
    re.compile(rb'sk-[A-Za-z0-9_-]{20,}'),
    re.compile(rb'(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}'),
    re.compile(rb'AKIA[A-Z0-9]{16}'),
    re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(rb'https?://[^\s"\'<>]+/rest/\d+/[A-Za-z0-9]{8,}'),
]


def sensitive_path(path: str) -> bool:
    name = PurePosixPath(path).name
    return ((name == '.env' or name.startswith('.env.')) and name != '.env.example'
            or name in ('id_rsa', 'id_ed25519')
            or name.endswith(('.pem', '.key', '.p12', '.pfx')))


def scan(revision: str | None = None) -> list[str]:
    args = ['git', 'ls-tree', '-rz', revision] if revision else ['git', 'ls-files', '--stage', '-z']
    entries = subprocess.check_output(args).split(b'\0')
    failures = []
    for entry in entries:
        if not entry:
            continue
        metadata, raw_path = entry.split(b'\t', 1)
        path = raw_path.decode('utf-8', errors='replace')
        blob = metadata.split()[2 if revision else 1].decode()
        content = subprocess.check_output(['git', 'cat-file', 'blob', blob])
        if sensitive_path(path) or any(pattern.search(content) for pattern in PATTERNS):
            failures.append(path)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', metavar='REV', help='Scan all commits reachable from REV')
    args = parser.parse_args()
    revisions = subprocess.check_output(['git', 'rev-list', args.history]).decode().splitlines() if args.history else [None]
    found = False
    for revision in revisions:
        for path in scan(revision):
            print(f'Secret candidate: {revision or "index"}:{path} (value hidden)')
            found = True
    if not found:
        print('Secret scan passed.')
    return int(found)


if __name__ == '__main__':
    raise SystemExit(main())
