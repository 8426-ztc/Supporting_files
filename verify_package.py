from pathlib import Path
import csv, hashlib, sys
root=Path(__file__).resolve().parent
bad=[]
with (root/'MANIFEST_SHA256.csv').open(encoding='utf-8-sig',newline='') as f:
    rows=list(csv.DictReader(f))
for row in rows:
    p=root/row['relative_path']
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
        bad.append(row['relative_path'])
expected={r['relative_path'] for r in rows}|{'MANIFEST_SHA256.csv'}
extra=[p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.relative_to(root).as_posix() not in expected]
print(f'Checked {len(rows)} files; mismatches/missing: {len(bad)}; extra files: {len(extra)}')
for name in bad+extra:print(name)
sys.exit(bool(bad or extra))
