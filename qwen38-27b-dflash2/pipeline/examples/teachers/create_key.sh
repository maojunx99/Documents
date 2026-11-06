#!/usr/bin/env bash
# Run once; securely copy this private file to the trainer and selected teacher hosts.
set -euo pipefail
PYTHON="${PYTHON:-python}"
KEY_FILE="${KEY_FILE:-/run/secrets/dflash2_teacher.key}"
"$PYTHON" - "$KEY_FILE" <<'PY'
import os,pathlib,secrets,sys
p=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True)
fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as f:f.write(secrets.token_hex(32)+'\n')
print('Created private key file:',p)
PY
