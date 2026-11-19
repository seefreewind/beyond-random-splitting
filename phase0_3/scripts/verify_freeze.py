from pathlib import Path
import csv,hashlib
ROOT=Path(__file__).resolve().parents[2]
with (ROOT/'FROZEN_INPUTS_SHA256.tsv').open() as stream:
 for row in csv.DictReader(stream,delimiter='\t'):
  path=ROOT/row['path']
  assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],row['path']
print('Public frozen inputs verified.')
