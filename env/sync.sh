#!/usr/bin/env bash
# Pull rewards.jsonl + sample PNGs/JS from the Colab VM into ./out
set -e
cd "$(dirname "$0")/.."
mkdir -p out/samples
colab --auth=adc download -s ${SESSION:-rlpaint2} /content/out/rewards.jsonl out/rewards.jsonl >/dev/null 2>&1 || true
python3 - <<'PY' > temp/sample_list.txt
import subprocess,json
r=subprocess.run("colab --auth=adc ls -s ${SESSION:-rlpaint2} /content/out/samples",shell=True,capture_output=True,text=True).stdout
for line in r.splitlines():
    t=line.strip().split()[-1] if line.strip() else ''
    if t.endswith('.png') or t.endswith('.js'): print(t)
PY
while read f; do [ -f "out/samples/$f" ] || colab --auth=adc download -s ${SESSION:-rlpaint2} "/content/out/samples/$f" "out/samples/$f" >/dev/null 2>&1; done < temp/sample_list.txt
echo "synced $(ls out/samples | wc -l | tr -d ' ') files"; tail -3 out/rewards.jsonl 2>/dev/null
