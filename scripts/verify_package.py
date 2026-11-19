from pathlib import Path
import csv,hashlib
ROOT=Path(__file__).resolve().parents[1]
with (ROOT/'MANIFEST_SHA256.tsv').open() as stream:
 for row in csv.DictReader(stream,delimiter='\t'):
  assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
print('All distributed files verified.')
