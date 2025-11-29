#!/usr/bin/env python3
import re
import sys
from pathlib import Path

SKIP_DIRS = {
    '.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.idea', '.vscode',
    'Kapitanov_MRRC/venv'
}
SKIP_EXT = {'.png', '.jpg', '.jpeg', '.gif', '.pdf', '.bin', '.pyc', '.pyo', '.so', '.dylib'}

PATTERNS = {
    'email': re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    'aws_access_key': re.compile(r"AKIA[0-9A-Z]{16}"),
    'github_token': re.compile(r"ghp_[0-9A-Za-z]{36}"),
    'generic_token': re.compile(r"(?i)(api[_-]?key|token|secret)[\s:=]{1,20}[\"']?[A-Za-z0-9._-]{10,}"),
    'private_key_marker': re.compile(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
    'ssh_pubkey': re.compile(r"ssh-(rsa|ed25519)\s+[A-Za-z0-9/+]+=+"),
    'path_user_home': re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    'phone_like': re.compile(r"\b\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"),
}

def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & SKIP_DIRS:
        return True
    if path.suffix.lower() in SKIP_EXT:
        return True
    return False

def scan(root: Path) -> int:
    findings = 0
    for p in root.rglob('*'):
        if p.is_dir() and p.name in SKIP_DIRS:
            continue
        if not p.is_file():
            continue
        if should_skip(p):
            continue
        try:
            text = p.read_text(errors='ignore')
        except Exception:
            continue
        for name, rx in PATTERNS.items():
            for m in rx.finditer(text):
                findings += 1
                snippet = text[max(0, m.start()-30):m.end()+30].replace('\n', ' ')
                print(f"[PII?] {name:>18} | {p} : {snippet}")
    return findings

def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    total = scan(root.resolve())
    if total == 0:
        print("No PII/secrets detected by simple scanner.")
        return 0
    print(f"\nTotal suspicious matches: {total}")
    return 1

if __name__ == '__main__':
    raise SystemExit(main())
