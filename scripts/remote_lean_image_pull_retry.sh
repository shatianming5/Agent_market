#!/usr/bin/env bash
set -u

ROOT=${ROOT:-/mnt/SSD_4TB/zechuan}
REPO=${REPO:-$ROOT/Agent_market}
XR=${XR:-$ROOT/codex_tools/xray/bin/xray}
SUB=${SUB:-$ROOT/proxy_subscription.raw}
OCI=${OCI:-$ROOT/quantconnect-lean-oci-20260501}
LOG=${LOG:-$ROOT/lean_image_pull_retry.log}
IMAGE=${IMAGE:-docker.io/quantconnect/lean:latest}

INDICES=(${INDICES:-140 141 142 143 144 145 146 149 151 152 153 154 155 156 157 160 161 162 163 164 165 35 40 58 59 12 22 25 30 33 38 41 43 44})

mkdir -p "$OCI"
cd "$REPO" || exit 1
echo "RESTART_FAST $(date -Is)" >> "$LOG"

for attempt in $(seq 1 300); do
  idx=${INDICES[$(((attempt - 1) % ${#INDICES[@]}))]}
  hp=$((18000 + attempt % 1000))
  sp=$((19000 + attempt % 1000))
  cfg=$ROOT/codex_tools/xray/etc/pull-fast-${attempt}-${idx}.json
  echo "ATTEMPT $attempt idx=$idx hp=$hp $(date -Is)" >> "$LOG"

  python3 scripts/xray_sub_to_config.py \
    --subscription-file "$SUB" \
    --index "$idx" \
    --socks-port "$sp" \
    --http-port "$hp" \
    --output "$cfg" >> "$LOG" 2>&1 || continue

  "$XR" run -test -config "$cfg" >> "$LOG" 2>&1 || continue
  "$XR" run -config "$cfg" >> "$ROOT/xray_pull_attempt.log" 2>&1 &
  xpid=$!
  sleep 1

  HTTP_PROXY=http://127.0.0.1:$hp HTTPS_PROXY=http://127.0.0.1:$hp \
    timeout 45 skopeo inspect --override-os linux --override-arch amd64 docker://$IMAGE \
    >/tmp/lean-inspect-${attempt}.json 2>>"$LOG"
  inspect_rc=$?
  if [ "$inspect_rc" != "0" ]; then
    echo "INSPECT_FAIL $attempt idx=$idx rc=$inspect_rc $(date -Is)" >> "$LOG"
    kill "$xpid" >/dev/null 2>&1 || true
    wait "$xpid" >/dev/null 2>&1 || true
    sleep 2
    continue
  fi
  echo "INSPECT_OK $attempt idx=$idx $(date -Is)" >> "$LOG"

  HTTP_PROXY=http://127.0.0.1:$hp HTTPS_PROXY=http://127.0.0.1:$hp \
    timeout 1800 skopeo copy --retry-times 3 --override-os linux --override-arch amd64 \
    docker://$IMAGE oci:"$OCI":latest >> "$LOG" 2>&1
  rc=$?
  kill "$xpid" >/dev/null 2>&1 || true
  wait "$xpid" >/dev/null 2>&1 || true
  du -sh "$OCI" >> "$LOG" 2>&1 || true
  echo "ATTEMPT_DONE $attempt idx=$idx rc=$rc $(date -Is)" >> "$LOG"
  if [ "$rc" = "0" ]; then
    echo "DONE $(date -Is)" >> "$LOG"
    exit 0
  fi
  sleep 3
done

echo "FAILED $(date -Is)" >> "$LOG"
exit 1
