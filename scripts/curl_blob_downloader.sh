#!/usr/bin/env bash
# Downloads missing OCI blobs for quantconnect/lean using curl -C - (resume support)
set -uo pipefail

ROOT=${ROOT:-/mnt/SSD_4TB/zechuan}
OCI=${OCI:-$ROOT/quantconnect-lean-oci-20260501}
BLOBS=$OCI/blobs/sha256
HP=${HP:-18008}   # Xray HTTP proxy port
LOG=${LOG:-$ROOT/curl_blob_dl.log}
XR=$ROOT/codex_tools/xray/bin/xray
SUB=$ROOT/proxy_subscription.raw
INDICES=(140 141 142 143 144 145 146 149 151 152 153 154 155 156 157 160 161 162 163 164 165 35 40 58 59 12 22 25 30 33 38 41 43 44)

MISSING_BLOBS=(
  ac85f4ab528160e4ce73ad61b594c52a7029e2840df5b58b97edabf3beec2946
  f0009b7b7ef1ae7c4186e90afe5a0a0fcfe245d8f6202c85cc893f9d626a65b4
  1d82a971bb755c2ba4ce1bef85f4995fc16ef131369c4bec902429524d323463
  892dce12f8ad78fd449ea8d37641f19f2a0fcc992083d0e515723cbde1e5c21c
  c91556d7d9dfb8eb83f6fe0202f93f8236bdb0938b45eb5e9b317addbca3f611
  befc0268797043e6db9dc4e2778aaaf2d21be873e8f9c452ff28002eb393001b
  80083f0f1b07b623049b495d27ef30926ea71bc800b20e27f973ace86a309d05
  d2275ceaf082e9a2a88f5d001c5c32b630feecb60e6282cde3d20c85236a3576
  557102c7b457d785e585ea417ac166243d1b55d58add7754f4a330c0c86a859f
  36fa0aff5b49a35da404eb43f2016926aa4b576c5406ba49c6343e23f2273bcf
  b1e58c74b9dadfd3de1cf14635d0389c74753d3ea90b74f7ae2c8deaf3237002
  f80fad9c187507536b79e563ba0fcdbfd8d71f9237a3f665047d3619fe1bead5
  61c230e61fb7188e8a30fdff16b0e225c07e86eced215bd2cb9614b0133086af
  3dc65eb25e0bfee5341016747f860edb62ad7f8f8cfbdebf324e72abe25ac363
  051ad4c8721ccde503ad928f005b224de40ccae98e9f8649f7ddff81ab0f2a68
  8fc8718b09b37f051ca4d8ac372730485aaea47c254e9dea2059197870bca757
  33cf00acb09e20ed22c66af2511b66060f2c4348371fa3e05f70a97861322577
  bee57c5aa4948c621db06e49b4964a386833f0e91d49b7bbc097d66a59236b28
  1d312a07b404346655fd03e7276c5a25973e94497114ab9ede36516bb1336b09
  1dc7e567fefd9a11ab5ef6fdc7869a47111a4fb0013f6eb8bfc02b0be91a1d4b
  849173dc2734b2e4537ba4fb471a2f467ede2e9e24369124994f8914e35c0c7a
  77299830b481d936627f2a5b191fd9ff51b359744c16360d9677f350e9258314
  e2945ea57b6e06150bcc740f7c991e4809eb84698975ec7a1ebc32e97b541762
  c1886f63871a4bbe5f32f6b6e4c8bc7c023f135bb34975e2c7fbf21070cd8939
  4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1
)

mkdir -p "$BLOBS"
echo "START $(date -Is)" | tee -a "$LOG"

XPID=""
CURRENT_HP=$HP

start_xray() {
  local idx=$1 hp=$2 sp=$((hp + 1000))
  local cfg="$ROOT/codex_tools/xray/etc/curl-dl-${hp}-${idx}.json"
  python3 "$ROOT/Agent_market/scripts/xray_sub_to_config.py" \
    --subscription-file "$SUB" --index "$idx" \
    --socks-port "$sp" --http-port "$hp" --output "$cfg" >> "$LOG" 2>&1 || return 1
  "$XR" run -test -config "$cfg" >> "$LOG" 2>&1 || return 1
  "$XR" run -config "$cfg" >> "$ROOT/xray_curl_dl.log" 2>&1 &
  XPID=$!
  sleep 2
  HTTP_PROXY=http://127.0.0.1:$hp HTTPS_PROXY=http://127.0.0.1:$hp \
    timeout 30 curl -s -o /dev/null -w "%{http_code}" \
    "https://auth.docker.io/token?service=registry.docker.io&scope=repository:quantconnect/lean:pull" \
    | grep -q "200" || return 1
  CURRENT_HP=$hp
  echo "XRAY_OK idx=$idx hp=$hp $(date -Is)" | tee -a "$LOG"
}

kill_xray() {
  [ -n "$XPID" ] && kill "$XPID" 2>/dev/null || true
  wait "$XPID" 2>/dev/null || true
  XPID=""
}

get_token() {
  curl -s -x "http://127.0.0.1:$CURRENT_HP" \
    "https://auth.docker.io/token?service=registry.docker.io&scope=repository:quantconnect/lean:pull" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['token'])" 2>/dev/null
}

# Try to use existing Xray on HP first, else find a new node
TOKEN=$(get_token 2>/dev/null || true)
if [ -z "$TOKEN" ]; then
  echo "NEED_NEW_NODE existing hp=$HP not working" | tee -a "$LOG"
  kill_xray
  for attempt in $(seq 1 50); do
    idx=${INDICES[$(((attempt - 1) % ${#INDICES[@]}))]}
    hp=$((20000 + attempt))
    kill_xray
    start_xray "$idx" "$hp" && TOKEN=$(get_token) && [ -n "$TOKEN" ] && break || true
    sleep 2
  done
fi

if [ -z "$TOKEN" ]; then
  echo "FATAL: cannot get Docker token" | tee -a "$LOG"; exit 1
fi
TOKEN_TIME=$(date +%s)
echo "GOT_TOKEN $(date -Is)" | tee -a "$LOG"

refresh_token_if_needed() {
  local now; now=$(date +%s)
  if [ $((now - TOKEN_TIME)) -gt 480 ]; then
    TOKEN=$(get_token 2>/dev/null || true)
    if [ -z "$TOKEN" ]; then
      # switch node
      kill_xray
      for attempt in $(seq 1 50); do
        idx=${INDICES[$(((attempt - 1) % ${#INDICES[@]}))]}
        hp=$((21000 + attempt))
        start_xray "$idx" "$hp" && TOKEN=$(get_token) && [ -n "$TOKEN" ] && break || true
        sleep 2
      done
    fi
    TOKEN_TIME=$(date +%s)
    echo "TOKEN_REFRESHED $(date -Is)" | tee -a "$LOG"
  fi
}

DONE=0
TOTAL=${#MISSING_BLOBS[@]}

for hash in "${MISSING_BLOBS[@]}"; do
  dest="$BLOBS/$hash"
  partial="$dest.partial"

  if [ -f "$dest" ]; then
    actual=$(sha256sum "$dest" | awk '{print $1}')
    if [ "$actual" = "$hash" ]; then
      echo "SKIP $hash" | tee -a "$LOG"
      DONE=$((DONE + 1))
      continue
    else
      echo "CORRUPT removing $hash" | tee -a "$LOG"
      rm -f "$dest"
    fi
  fi

  echo "DOWNLOAD $hash ($(du -sh "$partial" 2>/dev/null | awk '{print $1}' || echo '0') partial) $(date -Is)" | tee -a "$LOG"

  success=0
  for retry in $(seq 1 200); do
    refresh_token_if_needed

    curl -C - -L \
      -H "Authorization: Bearer $TOKEN" \
      -x "http://127.0.0.1:$CURRENT_HP" \
      --connect-timeout 30 --max-time 1800 \
      --retry 0 \
      -o "$partial" \
      "https://registry-1.docker.io/v2/quantconnect/lean/blobs/sha256:$hash" \
      >> "$LOG" 2>&1
    rc=$?

    if [ $rc -eq 33 ]; then
      echo "NO_RESUME $hash removing partial" | tee -a "$LOG"
      rm -f "$partial"; continue
    fi

    if [ $rc -ne 0 ] && [ $rc -ne 22 ]; then
      echo "CURL_RC=$rc $hash retry=$retry switching node" | tee -a "$LOG"
      TOKEN=$(get_token 2>/dev/null || true)
      if [ -z "$TOKEN" ]; then
        kill_xray
        for a in $(seq 1 20); do
          idx=${INDICES[$(((retry * 20 + a - 1) % ${#INDICES[@]}))]}
          hp=$((22000 + retry * 20 + a))
          start_xray "$idx" "$hp" && TOKEN=$(get_token) && [ -n "$TOKEN" ] && break || true
          sleep 2
        done
        TOKEN_TIME=$(date +%s)
      fi
      sleep 3
      continue
    fi

    if [ -f "$partial" ]; then
      actual=$(sha256sum "$partial" | awk '{print $1}')
      if [ "$actual" = "$hash" ]; then
        mv "$partial" "$dest"
        echo "OK $hash $(du -sh "$dest" | awk '{print $1}') $(date -Is)" | tee -a "$LOG"
        success=1
        break
      else
        echo "HASH_MISMATCH $hash actual=$actual retry=$retry" | tee -a "$LOG"
        rm -f "$partial"
        TOKEN=$(get_token 2>/dev/null || true)
        TOKEN_TIME=$(date +%s)
      fi
    fi
    sleep 2
  done

  if [ $success -eq 1 ]; then
    DONE=$((DONE + 1))
    echo "PROGRESS $DONE/$TOTAL $(date -Is)" | tee -a "$LOG"
  else
    echo "FAILED $hash after exhausting retries" | tee -a "$LOG"
  fi
done

kill_xray

echo "BLOBS_PHASE_DONE $DONE/$TOTAL $(date -Is)" | tee -a "$LOG"

# Final skopeo pass to complete index.json and manifests
echo "FINAL_SKOPEO_START $(date -Is)" | tee -a "$LOG"
for attempt in $(seq 1 30); do
  idx=${INDICES[$(((attempt - 1) % ${#INDICES[@]}))]}
  hp=$((23000 + attempt))
  cfg="$ROOT/codex_tools/xray/etc/final-skopeo-${hp}-${idx}.json"
  python3 "$ROOT/Agent_market/scripts/xray_sub_to_config.py" \
    --subscription-file "$SUB" --index "$idx" \
    --socks-port $((hp+1000)) --http-port "$hp" --output "$cfg" >> "$LOG" 2>&1 || continue
  "$XR" run -test -config "$cfg" >> "$LOG" 2>&1 || continue
  "$XR" run -config "$cfg" >> "$ROOT/xray_final_skopeo.log" 2>&1 &
  XPID=$!
  sleep 2
  HTTP_PROXY=http://127.0.0.1:$hp HTTPS_PROXY=http://127.0.0.1:$hp \
    timeout 30 skopeo inspect --override-os linux --override-arch amd64 \
    docker://docker.io/quantconnect/lean:latest > /dev/null 2>&1 || { kill $XPID 2>/dev/null; continue; }
  echo "FINAL_NODE_OK idx=$idx hp=$hp $(date -Is)" | tee -a "$LOG"
  HTTP_PROXY=http://127.0.0.1:$hp HTTPS_PROXY=http://127.0.0.1:$hp \
    timeout 600 skopeo copy --retry-times 5 --override-os linux --override-arch amd64 \
    docker://docker.io/quantconnect/lean:latest \
    oci:"$OCI":latest >> "$LOG" 2>&1
  rc=$?
  kill $XPID 2>/dev/null || true
  wait $XPID 2>/dev/null || true
  XPID=""
  if [ $rc -eq 0 ]; then
    echo "SKOPEO_DONE $(date -Is)" | tee -a "$LOG"
    break
  fi
done

if [ -f "$OCI/index.json" ]; then
  echo "OCI_COMPLETE loading into Docker $(date -Is)" | tee -a "$LOG"
  skopeo copy oci:"$OCI":latest docker-daemon:quantconnect/lean:latest >> "$LOG" 2>&1
  docker images quantconnect/lean >> "$LOG" 2>&1
  echo "DOCKER_LOAD_DONE $(date -Is)" | tee -a "$LOG"
else
  echo "WARN index.json not found after final skopeo" | tee -a "$LOG"
fi
