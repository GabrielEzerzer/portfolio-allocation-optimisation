#!/usr/bin/env python3
"""
Secret scanning script to prevent accidental commits of API keys and tokens.
Run as: python scripts/check_secrets.py
Exit code 0 = no secrets found, 1 = secrets detected
"""

import re
import subprocess
import sys
from pathlib import Path

# Patterns that commonly indicate secrets
SECRET_PATTERNS = [
    # API Keys (generic patterns)
    (r'(?i)["\']?api[_-]?key["\']?\s*[:=]\s*["\'][a-zA-Z0-9]{16,}["\']', 'API Key'),
    (r'(?i)["\']?apikey["\']?\s*[:=]\s*["\'][a-zA-Z0-9]{16,}["\']', 'API Key'),
    
    # Alpha Vantage specific (16 chars, usually uppercase, avoiding common words)
    # Added boundaries and excluded mixed case words that look like class names
    (r'\b[A-Z0-9]{16}\b', 'Possible Alpha Vantage Key'),
    
    # AWS
    (r'\bAKIA[0-9A-Z]{16}\b', 'AWS Access Key'),
    (r'(?i)["\']?aws[_-]?secret["\']?\s*[:=]\s*["\'][a-zA-Z0-9/+]{40}["\']', 'AWS Secret'),
    
    # Generic tokens
    (r'(?i)["\']?token["\']?\s*[:=]\s*["\'][a-zA-Z0-9_-]{20,}["\']', 'Token'),
    (r'(?i)["\']?secret["\']?\s*[:=]\s*["\'][a-zA-Z0-9_-]{20,}["\']', 'Secret'),
    
    # Private keys
    (r'-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----', 'Private Key'),
    
    # Bearer tokens
    (r'Bearer\s+[a-zA-Z0-9_-]{20,}', 'Bearer Token'),
]

# Files/patterns to skip
SKIP_PATTERNS = [
    r'\.git/',
    r'__pycache__/',
    r'\.pyc$',
    r'node_modules/',
    r'\.env\.example$',
    r'settings\.example\.yaml$',
    r'check_secrets\.py$',  # Skip this file itself
    r'\.md$',  # Skip markdown docs (they have examples)
]

# Known safe patterns (placeholders)
SAFE_PATTERNS = [
    r'YOUR_API_KEY_HERE',
    r'YOUR_KEY_HERE',
    r'\$\{[^}]+\}',  # Environment variable placeholders
    r'<[^>]+>',  # Placeholder tags
    r'xxx+',  # Redacted values
    r'placeholder',
]


def should_skip_file(filepath: str) -> bool:
    """Check if file should be skipped."""
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, filepath, re.IGNORECASE):
            return True
    return False


def is_safe_value(line: str) -> bool:
    """Check if the matched line contains a safe placeholder."""
    for pattern in SAFE_PATTERNS:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    return False


def get_tracked_files() -> list[str]:
    """Get list of git-tracked files."""
    try:
        result = subprocess.run(
            ['git', 'ls-files'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split('\n')
    except subprocess.CalledProcessError:
        # Fallback: scan all files in current directory
        return [str(p) for p in Path('.').rglob('*') if p.is_file()]


def scan_file(filepath: str) -> list[tuple[int, str, str]]:
    """Scan a single file for secrets. Returns list of (line_num, pattern_name, line_content)."""
    findings = []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                # Skip if contains safe placeholder
                if is_safe_value(line):
                    continue
                
                for pattern, name in SECRET_PATTERNS:
                    # Note: case sensitivity is now handled by the pattern itself via (?i) if needed
                    if re.search(pattern, line):
                        findings.append((line_num, name, line.strip()[:100]))
                        break  # One finding per line is enough
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
    
    return findings


def main() -> int:
    """Main entry point. Returns exit code."""
    print("Scanning for secrets in tracked files...")
    
    files = get_tracked_files()
    total_findings = 0
    files_with_secrets = []
    
    for filepath in files:
        if not filepath or should_skip_file(filepath):
            continue
        
        findings = scan_file(filepath)
        if findings:
            files_with_secrets.append(filepath)
            total_findings += len(findings)
            print(f"\n[FAIL] {filepath}:")
            for line_num, pattern_name, content in findings:
                print(f"   Line {line_num} [{pattern_name}]: {content}")
    
    print()
    if total_findings > 0:
        print(f"FAILED: Found {total_findings} potential secret(s) in {len(files_with_secrets)} file(s)")
        print("   Please remove secrets and use environment variables or .env files instead.")
        return 1
    else:
        print(f"PASSED: No secrets detected in {len(files)} files")
        return 0


if __name__ == '__main__':
    sys.exit(main())
