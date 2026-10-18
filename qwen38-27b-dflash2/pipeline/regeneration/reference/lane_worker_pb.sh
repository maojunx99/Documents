#!/bin/bash
# Publication change: workspace paths anonymized; see pipeline/PROVENANCE.md.
# lane_worker_pb.sh TAG HOSTPORT — pb860k regen lane: ShareGPT conversations input
# Same protocol as 27B regen: temp 1.0/top_p 0.95/top_k 20/presence_penalty 1.5/xhigh/8192, concurrency 34
TAG=$1; HOSTPORT=$2
W=/workspace/dflash2_npu
MODEL=/models/Qwen3.8-27B
IN=$W/data/pb800k/shards
BASE=$W/output/regen27b_pb860k
OUT=$BASE/shards
Q=$BASE/queue
SAMPLING='{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}'
LOG=$BASE/logs/lane_${TAG}.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
mkdir -p $OUT $Q/todo $Q/running $Q/done $Q/incomplete $Q/claims $BASE/logs

log "lane start, endpoint $HOSTPORT"
while true; do
    curl -sf -m 3 "http://${HOSTPORT}/health" > /dev/null 2>&1 || { log "server $HOSTPORT unhealthy, waiting"; sleep 60; continue; }
    item=""
    for f in $(ls "$Q/todo" 2>/dev/null); do
        if mkdir "$Q/claims/$f" 2>/dev/null; then
            mv "$Q/todo/$f" "$Q/running/${TAG}__$f"
            item=$f; break
        fi
    done
    if [ -z "$item" ]; then
        [ -f "$Q/STOP" ] && { log "STOP, lane done"; exit 0; }
        sleep 120; continue
    fi
    base=${item#*_}
    src=${base%_*}; idx=${base##*_}
    in=$IN/${src}_${idx}.jsonl
    out=$OUT/${src}_${idx}_out.jsonl
    if [ ! -f "$in" ]; then
        log "INPUT MISSING $base"
        mv "$Q/running/${TAG}__$item" "$Q/incomplete/${TAG}__$item"
        continue
    fi
    n=$(wc -l < "$in"); dn=0
    [ -f "$out" ] && dn=$(wc -l < "$out")
    log "START $base $dn/$n"
    speculators regenerate-responses \
        --endpoint "http://${HOSTPORT}/v1/chat/completions" \
        --model "$MODEL" \
        --dataset "$in" \
        --concurrency 34 \
        --max-tokens 8192 \
        --sampling-params "$SAMPLING" \
        --reasoning-effort-cycle xhigh \
        --resume \
        --outfile "$out" >> $BASE/logs/client_${TAG}.log 2>&1
    dn=$(wc -l < "$out" 2>/dev/null || echo 0)
    if [ "$n" -gt 0 ] && [ "$dn" -ge "$n" ]; then
        mv "$Q/running/${TAG}__$item" "$Q/done/${TAG}__$item"
        log "DONE $base $dn/$n"
    else
        mv "$Q/running/${TAG}__$item" "$Q/incomplete/${TAG}__$item"
        log "INCOMPLETE $base $dn/$n"
    fi
done
