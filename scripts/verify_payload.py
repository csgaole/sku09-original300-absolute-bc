from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parents[1]
m=json.loads((root/'migration_manifest.json').read_text())
for row in m['files']:
 p=root/row['path'];assert p.stat().st_size==row['bytes'],p
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 assert h.hexdigest()==row['sha256'],p
assert len(list((root/'data/episodes').glob('*.h5')))==300
print('Verified',len(m['files']),'files and 300 demonstrations')
