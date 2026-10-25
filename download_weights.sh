#!/bin/bash
# =============================================================================
# Download the pretrained weights used by CAPOS.
#
# Layout (under pretrained_weights/ at the repository root by default):
#
#   esm-s/
#     esm2_t33_650M_UR50D.pt     official fair-esm ESM-2 650M base
#     esm_650m_s.pth             ESM-S (TorchDrug state_dict)
#   saprot/
#     SaProt_650M_PDB/           HF westlake-repl/SaProt_650M_PDB
#     SaProt/                    code snapshot (needed by extract_saprot.py)
#   prott5/
#     prot_t5_xl_half_uniref50-enc/   HF Rostlab checkpoint (CAPOS-seq)
#   chai1/
#     conformers_v1.apkl
#     models_v2/*.pt
#     esm/                       HF cache for facebook/esm2_t36_3B_UR50D
#   foldseek/
#     bin/foldseek               built from FOLDSEEK_COMMIT
#     data/                      3Di matrices shipped with that commit
#
# Downloads run in parallel (one background job per component). Every local
# file is checked against the SHA256 of the version used to produce the CAPOS
# results. HF snapshots are pinned by revision.
#
# Usage:
#   ./download_weights.sh
#   ./download_weights.sh --dest /path/to/pretrained_weights --jobs 4
#   ./download_weights.sh --verify-only
#   ./download_weights.sh --components esm-s,saprot,prott5,chai1
#   ./download_weights.sh --seed-esms ... --seed-saprot ... \
#        --seed-prott5 ... --seed-chai ...
#
# --seed-* copies already-verified local assets instead of downloading them.
# The full-tier image already ships Foldseek, so the image deployment can omit
# the source-built foldseek component with --components.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${CAPOS_PRETRAINED_DIR:-$ROOT/pretrained_weights}"
JOBS="${CAPOS_DOWNLOAD_JOBS:-4}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
COMPONENTS="esm-s,saprot,prott5,chai1,foldseek"
VERIFY_ONLY=0
FORCE_DOWNLOAD=0
SEED_ESMS=""
SEED_SAPROT=""
SEED_PROTT5=""
SEED_CHAI=""

HF_BASE="${HF_ENDPOINT%/}"
CHAI_BASE="https://chaiassets.com/chai1-inference-depencencies"

# Versions / hashes of the assets actually used for the CAPOS results.
SHA_ESM2="ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c"
SHA_ESMS="da99ae0f52f4d7dccc57761782c438d95ecc6095ee48944c191b0844a000f867"
SHA_SAPROT="f37c51299a3ee31ead2823b835300e5d12f60453a2f69d2b977aa2a36b99641a"
SHA_PROTT5="7f51ba885541c7dc569d46b796af57cc7a2ba7945107dced4f19d1b5ec091157"
SAPROT_REV="6f7468fdd0cdffe7e780bd4f57fa291df6b01867"
PROTT5_REV="94a6abc029ae13029317b140b7424e012bf8dfbf"
CHAI_ESM_REV="476b639933c8baad5ad09a60ac1a87f987b656fc"

# Foldseek commit used for the paper. No prebuilt binary exists for this
# commit (the newest tagged release, 10-941cd33, points to 941cd33, which is
# 197 commits behind), so this commit is always built from source.
FOLDSEEK_COMMIT="c9fa0e8288f46c640cfd7f5102b5d741dfaf45dd"

CHAI_ASSETS=(
  "conformers_v1.apkl f2161256b565bbd84198da8b3e3d3978ae2f179f778d0995cd135c4c0c57c41f"
  "models_v2/feature_embedding.pt 58600fb282827ee105b4bf33845c2eb2dce2a7d2ba7582951f76c9290c66c019"
  "models_v2/bond_loss_input_proj.pt 98f3b2e415d11e460a684cdbfc213ca80447308022b1953cc5326ad468948851"
  "models_v2/token_embedder.pt 60cada19fd9c77d8298131a00368d162b6bc7d3c836dd36c715db04d21f7bbde"
  "models_v2/trunk.pt b7a08e104455f76491088a5eeaea9bb95a08168c48969b0b892971346e51ccbc"
  "models_v2/diffusion_module.pt bd77cf4095ad6b8a4eae7e390d01d24be1ebd819922ca3f28e8744f9945c9cad"
  "models_v2/confidence_head.pt 82a4ac9f934fdd0e73870150f508cecbe010c1383234bdadcaa70d57323deb6b"
)

usage() {
  sed -n '2,37p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --components) COMPONENTS="$2"; shift 2 ;;
    --seed-esms) SEED_ESMS="$2"; shift 2 ;;
    --seed-saprot) SEED_SAPROT="$2"; shift 2 ;;
    --seed-prott5) SEED_PROTT5="$2"; shift 2 ;;
    --seed-chai) SEED_CHAI="$2"; shift 2 ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    --force-download) FORCE_DOWNLOAD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "!! unknown option: $1" >&2; exit 2 ;;
  esac
done

export HF_ENDPOINT
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

check_file() {
  local path="$1" want="$2" got
  if [ ! -f "$path" ]; then
    echo "MISSING $path" >&2
    return 1
  fi
  got="$(sha256 "$path")"
  if [ "$got" != "$want" ]; then
    echo "SHA256 MISMATCH $path" >&2
    echo "  got      $got" >&2
    echo "  expected $want" >&2
    return 1
  fi
  say "OK $path"
}

seed_file() {
  local src="$1" dst="$2" want="$3"
  [ "$FORCE_DOWNLOAD" = 0 ] || return 1
  [ -f "$src" ] || return 1
  mkdir -p "$(dirname "$dst")"
  cp -f "$src" "$dst"
  check_file "$dst" "$want"
}

fetch_url() {
  local url="$1" dst="$2" want="$3"
  mkdir -p "$(dirname "$dst")"
  if [ -f "$dst" ] && check_file "$dst" "$want" >/dev/null 2>&1; then
    say "exists $dst"
    return 0
  fi
  if [ -f "$dst" ]; then
    mv "$dst" "$dst.bad.$(date +%s)"
  fi
  say "downloading $url"
  curl -fL --retry 5 --retry-delay 5 -C - -o "$dst.part" "$url"
  mv "$dst.part" "$dst"
  check_file "$dst" "$want"
}

hf_local() {
  local repo="$1" rev="$2" dst="$3" jobs="$4"; shift 4
  python - "$repo" "$rev" "$dst" "$jobs" "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo, rev, dst, jobs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
patterns = sys.argv[5:] or None
print(snapshot_download(repo_id=repo, revision=rev, local_dir=dst,
                        allow_patterns=patterns, max_workers=jobs))
PY
}

hf_cache() {
  local repo="$1" rev="$2" cache="$3" jobs="$4"; shift 4
  python - "$repo" "$rev" "$cache" "$jobs" "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo, rev, cache, jobs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
patterns = sys.argv[5:] or None
print(snapshot_download(repo_id=repo, revision=rev, cache_dir=cache,
                        allow_patterns=patterns, max_workers=jobs))
PY
}

install_saprot_code() {
  local dst="$DEST/saprot/SaProt"
  local zip="$ROOT/LICENSE_packages/SaProt-main.zip"
  if [ -f "$dst/model/saprot/base.py" ] && [ -f "$dst/utils/foldseek_util.py" ]; then
    :
  elif [ -f "$zip" ]; then
    say "extracting SaProt code from $zip"
    local tmp
    tmp="$(mktemp -d)"
    unzip -q "$zip" -d "$tmp"
    rm -rf "$dst"
    mv "$tmp/SaProt-main" "$dst"
    rm -rf "$tmp"
  else
    say "cloning SaProt code at e91e4858"
    git clone --filter=blob:none https://github.com/westlake-repl/SaProt.git "$dst"
    git -C "$dst" checkout e91e4858b55944523f1f8d385f7b96a0d3d34c1d
  fi
  local got
  got="$(sha256 "$dst/model/saprot/base.py")"
  [ "$got" = "59d3d726e1776a8ebe2d7b0a45468010ee77c4e8aea78616116a50c04262bbef" ] || {
    echo "!! SaProt base.py code hash mismatch" >&2; return 1; }
  got="$(sha256 "$dst/utils/foldseek_util.py")"
  [ "$got" = "5905f0c29db41a0c89677412635c8e3a1886906a3ea7e963124d7218d4b45449" ] || {
    echo "!! SaProt foldseek_util.py code hash mismatch" >&2; return 1; }
  say "OK SaProt code $dst"
}

component_esms() {
  mkdir -p "$DEST/esm-s"
  if ! seed_file "$SEED_ESMS/esm2_t33_650M_UR50D.pt" \
       "$DEST/esm-s/esm2_t33_650M_UR50D.pt" "$SHA_ESM2"; then
    fetch_url "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt" \
      "$DEST/esm-s/esm2_t33_650M_UR50D.pt" "$SHA_ESM2"
  fi
  if ! seed_file "$SEED_ESMS/esm_650m_s.pth" \
       "$DEST/esm-s/esm_650m_s.pth" "$SHA_ESMS"; then
    fetch_url "$HF_BASE/Oxer11/ESM-S/resolve/main/esm_650m_s.pth" \
      "$DEST/esm-s/esm_650m_s.pth" "$SHA_ESMS"
  fi
}

component_saprot() {
  local dst="$DEST/saprot/SaProt_650M_PDB"
  mkdir -p "$dst"
  if [ "$FORCE_DOWNLOAD" = 0 ] && [ -n "$SEED_SAPROT" ] \
     && [ -f "$SEED_SAPROT/pytorch_model.bin" ]; then
    cp -Lf "$SEED_SAPROT/config.json" "$SEED_SAPROT/pytorch_model.bin" \
           "$SEED_SAPROT/special_tokens_map.json" \
           "$SEED_SAPROT/tokenizer_config.json" \
           "$SEED_SAPROT/vocab.txt" "$dst/"
  else
    hf_local westlake-repl/SaProt_650M_PDB "$SAPROT_REV" "$dst" "$JOBS" \
      config.json pytorch_model.bin special_tokens_map.json \
      tokenizer_config.json vocab.txt README.md
  fi
  check_file "$dst/pytorch_model.bin" "$SHA_SAPROT"
  install_saprot_code
}

component_prott5() {
  local dst="$DEST/prott5/prot_t5_xl_half_uniref50-enc"
  mkdir -p "$dst"
  if [ "$FORCE_DOWNLOAD" = 0 ] && [ -n "$SEED_PROTT5" ] \
     && [ -f "$SEED_PROTT5/pytorch_model.bin" ]; then
    cp -Lf "$SEED_PROTT5/config.json" "$SEED_PROTT5/pytorch_model.bin" \
           "$SEED_PROTT5/special_tokens_map.json" \
           "$SEED_PROTT5/spiece.model" \
           "$SEED_PROTT5/tokenizer_config.json" "$dst/"
  else
    hf_local Rostlab/prot_t5_xl_half_uniref50-enc "$PROTT5_REV" "$dst" "$JOBS" \
      config.json pytorch_model.bin special_tokens_map.json spiece.model \
      tokenizer_config.json
  fi
  check_file "$dst/pytorch_model.bin" "$SHA_PROTT5"
}

component_chai() {
  mkdir -p "$DEST/chai1"
  local rel sha
  for item in "${CHAI_ASSETS[@]}"; do
    read -r rel sha <<< "$item"
    local dst="$DEST/chai1/$rel"
    if [ "$FORCE_DOWNLOAD" = 0 ] && [ -n "$SEED_CHAI" ] \
       && [ -f "$SEED_CHAI/$rel" ]; then
      mkdir -p "$(dirname "$dst")"
      cp -f "$SEED_CHAI/$rel" "$dst"
      check_file "$dst" "$sha"
    else
      fetch_url "$CHAI_BASE/$rel" "$dst" "$sha"
    fi
  done

  mkdir -p "$DEST/chai1/esm"
  if [ "$FORCE_DOWNLOAD" = 0 ] && [ -n "$SEED_CHAI" ] \
     && [ -d "$SEED_CHAI/esm" ]; then
    say "seeding Chai-1 ESM cache from $SEED_CHAI/esm"
    cp -a "$SEED_CHAI/esm/." "$DEST/chai1/esm/"
  fi
  say "downloading/verifying Chai-1 ESM-2 3B (HF revision $CHAI_ESM_REV)"
  hf_cache facebook/esm2_t36_3B_UR50D "$CHAI_ESM_REV" "$DEST/chai1/esm" "$JOBS" \
    config.json 'pytorch_model*' special_tokens_map.json tokenizer_config.json \
    vocab.txt README.md

  # Some huggingface_hub versions populate snapshots/blobs but do not create
  # refs/main. Transformers' offline resolver needs that ref to map the model
  # id to the pinned snapshot, otherwise Chai-1 fails at
  # EsmTokenizer.from_pretrained() on an offline compute node.
  local repo_cache="$DEST/chai1/esm/models--facebook--esm2_t36_3B_UR50D"
  mkdir -p "$repo_cache/refs"
  printf '%s' "$CHAI_ESM_REV" > "$repo_cache/refs/main"

  local snap="$DEST/chai1/esm/models--facebook--esm2_t36_3B_UR50D/snapshots/$CHAI_ESM_REV"
  [ -f "$snap/config.json" ] || { echo "!! Chai ESM config missing: $snap" >&2; return 1; }
  if [ -f "$snap/pytorch_model.bin" ] || [ -f "$snap/model.safetensors" ] \
     || [ -f "$snap/pytorch_model.bin.index.json" ] \
     || [ -f "$snap/model.safetensors.index.json" ]; then
    if [ -f "$snap/pytorch_model.bin.index.json" ]; then
      compgen -G "$snap/pytorch_model-*.bin" >/dev/null || {
        echo "!! Chai ESM shards missing under $snap" >&2; return 1; }
    fi
  else
    echo "!! Chai ESM weights missing: $snap" >&2
    return 1
  fi
  say "OK Chai-1 ESM cache $snap"
}

foldseek_stamp_value() {
  printf 'source:%s' "$FOLDSEEK_COMMIT"
}

build_foldseek_source() {
  local dst="$1" bin="$2" stamp="$3"
  local src="$dst/src"
  local tar="/tmp/foldseek-${FOLDSEEK_COMMIT}.tar.gz.$$"
  local jobs
  jobs="$(nproc 2>/dev/null || echo "$JOBS")"

  local tool
  for tool in cmake make c++ curl cargo; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      cat >&2 <<EOF
!! Cannot build Foldseek ${FOLDSEEK_COMMIT}: '$tool' not found.
   No prebuilt binary is published for this commit. Install cmake, make,
   a C++ compiler, Rust/cargo, and the zlib/bzip2 headers.
   If you already have a c9fa0e8 build, place it at
   ${bin} and write "source:${FOLDSEEK_COMMIT}" to ${stamp}.
EOF
      return 1
    fi
  done

  rm -rf "$src"
  mkdir -p "$src"
  say "downloading Foldseek source ${FOLDSEEK_COMMIT}"
  curl -fL --retry 5 --retry-delay 5 -o "$tar" \
    "https://github.com/steineggerlab/foldseek/archive/${FOLDSEEK_COMMIT}.tar.gz"
  tar -xzf "$tar" -C "$src" --strip-components=1
  rm -f "$tar"

  say "building Foldseek ${FOLDSEEK_COMMIT} (jobs=${jobs})"
  if ! (
    cd "$src"
    mkdir -p build
    cd build
    cmake -DHAVE_AVX2=1 -DHAVE_MPI=0 -DHAVE_TESTS=0 \
          -DENABLE_PROSTT5=0 -DCMAKE_BUILD_TYPE=Release \
          -DVERSION_OVERRIDE=${FOLDSEEK_COMMIT} ..
    make -j"$jobs"
  ); then
    echo "!! Foldseek build failed; see the CMake/make output above." >&2
    rm -rf "$src"
    return 1
  fi

  mkdir -p "$(dirname "$bin")"
  install -m 0755 "$src/build/src/foldseek" "$bin"
  if [ -d "$src/data" ]; then
    rm -rf "$dst/data"
    cp -a "$src/data" "$dst/data"
  fi
  rm -rf "$src"
  printf '%s\n' "source:$FOLDSEEK_COMMIT" > "$stamp"

  local ver
  ver="$("$bin" version 2>&1)" || {
    echo "!! Foldseek binary not executable after build" >&2
    return 1
  }
  say "OK Foldseek $ver built from $FOLDSEEK_COMMIT"
}

component_foldseek() {
  local dst="$DEST/foldseek"
  local bin="$dst/bin/foldseek"
  local stamp="$dst/.capos_foldseek_version"
  local want
  want="$(foldseek_stamp_value)"

  mkdir -p "$dst"

  if [ -f "$bin" ] && [ -x "$bin" ] && [ -f "$stamp" ] \
     && [ "$(cat "$stamp")" = "$want" ]; then
    if "$bin" version >/dev/null 2>&1; then
      say "Foldseek already installed ($want): $bin"
      return 0
    fi
    say "Foldseek binary corrupted, rebuilding"
  elif [ -f "$bin" ]; then
    say "existing Foldseek has no matching provenance stamp; installing $want"
  fi

  build_foldseek_source "$dst" "$bin" "$stamp"
}

verify_esms() {
  check_file "$DEST/esm-s/esm2_t33_650M_UR50D.pt" "$SHA_ESM2"
  check_file "$DEST/esm-s/esm_650m_s.pth" "$SHA_ESMS"
}
verify_saprot() {
  check_file "$DEST/saprot/SaProt_650M_PDB/pytorch_model.bin" "$SHA_SAPROT"
  [ -f "$DEST/saprot/SaProt/model/saprot/base.py" ] || {
    echo "MISSING $DEST/saprot/SaProt/model/saprot/base.py" >&2; return 1; }
  [ -f "$DEST/saprot/SaProt/utils/foldseek_util.py" ] || {
    echo "MISSING $DEST/saprot/SaProt/utils/foldseek_util.py" >&2; return 1; }
  say "OK SaProt code"
}
verify_prott5() {
  check_file "$DEST/prott5/prot_t5_xl_half_uniref50-enc/pytorch_model.bin" "$SHA_PROTT5"
}
verify_chai() {
  local item rel sha
  for item in "${CHAI_ASSETS[@]}"; do
    read -r rel sha <<< "$item"
    check_file "$DEST/chai1/$rel" "$sha"
  done
  local repo_cache="$DEST/chai1/esm/models--facebook--esm2_t36_3B_UR50D"
  [ -f "$repo_cache/refs/main" ] || {
    echo "MISSING $repo_cache/refs/main" >&2; return 1; }
  [ "$(cat "$repo_cache/refs/main")" = "$CHAI_ESM_REV" ] || {
    echo "!! Chai-1 ESM ref mismatch: $(cat "$repo_cache/refs/main")" >&2
    return 1; }
  local snap="$DEST/chai1/esm/models--facebook--esm2_t36_3B_UR50D/snapshots/$CHAI_ESM_REV"
  [ -f "$snap/config.json" ] || { echo "MISSING $snap/config.json" >&2; return 1; }
  if [ -f "$snap/pytorch_model.bin" ] || [ -f "$snap/model.safetensors" ] \
     || [ -f "$snap/pytorch_model.bin.index.json" ] \
     || [ -f "$snap/model.safetensors.index.json" ]; then
    if [ -f "$snap/pytorch_model.bin.index.json" ]; then
      compgen -G "$snap/pytorch_model-*.bin" >/dev/null || {
        echo "MISSING $snap/pytorch_model-*.bin" >&2; return 1; }
    fi
  else
    echo "MISSING $snap/pytorch_model*" >&2
    return 1
  fi
  say "OK Chai-1 ESM cache"
}
verify_foldseek() {
  local bin="$DEST/foldseek/bin/foldseek"
  local stamp="$DEST/foldseek/.capos_foldseek_version"
  [ -f "$bin" ] || { echo "MISSING $bin" >&2; return 1; }
  [ -x "$bin" ] || { echo "NOT EXECUTABLE $bin" >&2; return 1; }
  "$bin" version >/dev/null 2>&1 || {
    echo "!! Foldseek binary not working" >&2; return 1; }
  local want
  want="$(foldseek_stamp_value)"
  if [ ! -f "$stamp" ] || [ "$(cat "$stamp")" != "$want" ]; then
    echo "!! Foldseek provenance mismatch: expected $want, found $(cat "$stamp" 2>/dev/null || echo none)" >&2
    return 1
  fi
  say "OK Foldseek ($want)"
}

run_component() {
  case "$1" in
    esm-s) component_esms ;;
    saprot) component_saprot ;;
    prott5) component_prott5 ;;
    chai1) component_chai ;;
    foldseek) component_foldseek ;;
    *) echo "!! unknown component $1" >&2; return 2 ;;
  esac
}

verify_component() {
  case "$1" in
    esm-s) verify_esms ;;
    saprot) verify_saprot ;;
    prott5) verify_prott5 ;;
    chai1) verify_chai ;;
    foldseek) verify_foldseek ;;
    *) echo "!! unknown component $1" >&2; return 2 ;;
  esac
}

IFS=',' read -r -a comps <<< "$COMPONENTS"
mkdir -p "$DEST/logs"

if ! python -c 'import huggingface_hub' >/dev/null 2>&1; then
  echo "!! huggingface_hub is not installed in the active environment" >&2
  echo "   pip install 'huggingface_hub>=0.23'" >&2
  exit 1
fi

if [ "$VERIFY_ONLY" = 1 ]; then
  fail=0
  for c in "${comps[@]}"; do
    verify_component "$c" || fail=1
  done
  [ "$fail" = 0 ] && say "ALL WEIGHTS VERIFIED" && exit 0
  exit 1
fi

say "destination: $DEST"
say "components : ${comps[*]}"
say "jobs       : $JOBS"
say "HF endpoint: $HF_ENDPOINT"

pids=()
for c in "${comps[@]}"; do
  ( run_component "$c" ) >"$DEST/logs/$c.log" 2>&1 &
  pids+=($!)
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "!! component failed: ${comps[$i]}" >&2
    tail -40 "$DEST/logs/${comps[$i]}.log" >&2 || true
    fail=1
  else
    say "component done: ${comps[$i]}"
  fi
done

[ "$fail" = 0 ] || exit 1

say "verifying all downloaded assets"
for c in "${comps[@]}"; do
  verify_component "$c" || fail=1
done
[ "$fail" = 0 ] || exit 1

{
  echo "$SHA_ESM2  esm-s/esm2_t33_650M_UR50D.pt"
  echo "$SHA_ESMS  esm-s/esm_650m_s.pth"
  echo "$SHA_SAPROT  saprot/SaProt_650M_PDB/pytorch_model.bin"
  echo "$SHA_PROTT5  prott5/prot_t5_xl_half_uniref50-enc/pytorch_model.bin"
  for item in "${CHAI_ASSETS[@]}"; do
    read -r rel sha <<< "$item"
    echo "$sha  chai1/$rel"
  done
} > "$DEST/checksums.sha256"

# Generate environment variable config file
cat > "$DEST/capos_env.sh" <<ENVEOF
#!/bin/bash
# CAPOS environment variables - source this file before running capos
# Generated by download_weights.sh on $(date)

export CAPOS_ESMS_MODEL=$DEST/esm-s
export CAPOS_ESMS_CKPT=$DEST/esm-s/esm_650m_s.pth
export CAPOS_SAPROT_MODEL=$DEST/saprot/SaProt_650M_PDB
export CAPOS_SAPROT_REPO=$DEST/saprot/SaProt
export CAPOS_PROTT5_MODEL=$DEST/prott5/prot_t5_xl_half_uniref50-enc
export CHAI_DOWNLOADS_DIR=$DEST/chai1
export CAPOS_FOLDSEEK_BIN=$DEST/foldseek/bin/foldseek

# HuggingFace mirror for China mainland users -- uncomment if huggingface.co
# is slow/blocked (official endpoint is used by default)
# export HF_ENDPOINT=https://hf-mirror.com

# HuggingFace cache directories
export HF_HOME=$DEST/chai1/esm
export TRANSFORMERS_CACHE=$DEST/chai1/esm
ENVEOF
chmod +x "$DEST/capos_env.sh"

cat <<EOF

All weights and tools are in: $DEST

Environment variables written to: $DEST/capos_env.sh
Source it with: source $DEST/capos_env.sh

Or export manually:
  export CAPOS_ESMS_MODEL=$DEST/esm-s
  export CAPOS_ESMS_CKPT=$DEST/esm-s/esm_650m_s.pth
  export CAPOS_SAPROT_MODEL=$DEST/saprot/SaProt_650M_PDB
  export CAPOS_SAPROT_REPO=$DEST/saprot/SaProt
  export CAPOS_PROTT5_MODEL=$DEST/prott5/prot_t5_xl_half_uniref50-enc
  export CHAI_DOWNLOADS_DIR=$DEST/chai1
  export CAPOS_FOLDSEEK_BIN=$DEST/foldseek/bin/foldseek

These weights remain under their upstream licences.
Check the upstream model cards before redistributing them.
EOF
