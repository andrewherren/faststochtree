#!/usr/bin/env bash
# bench/run_rmse_experiment.sh — M-dataset × K-chain RMSE comparison experiment
#
# For each (tag, dataset m, chain k):
#   data_seed  = BASE_DATA_SEED + m          — same across all tags for dataset m
#   model_seed = BASE_MODEL_SEED + m*K + k   — unique per chain
#
# This separates dataset variance from sampler variance, enabling a fair
# RMSE comparison across algorithm versions on identical problems.
#
# Output CSV columns differ from bench_results_mc.csv:
#   tag, timestamp, mode, n_train, p, trees, burnin, samples, threads,
#   dataset_id, chain_id, M, K, time_ms, rmse, sigma_post
#
# EXAMPLES
#   ./bench/run_rmse_experiment.sh --tags v1-naive-cpp,v3-quantized
#   ./bench/run_rmse_experiment.sh --tags v1-naive-cpp,v3-quantized --M 20 --K 5
#   ./bench/run_rmse_experiment.sh --quick

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BENCH_BUILDS="$REPO_DIR/.bench-builds"
FETCH_CACHE="$REPO_DIR/.cmake-fetch-cache"

# ── Defaults ──────────────────────────────────────────────────────────────────
N_VALS=(50000)
P_VALS=(50)
TREES=200
BURNIN=""
SAMPLES=""
MODE="bart"
THREADS=1
M=10
K=5
BASE_DATA_SEED=1234
BASE_MODEL_SEED=4321
TIMEOUT_S=0
SKIP_EXISTING=true
FORCE_REBUILD=false
TAGS=()
RESULTS_CSV="$SCRIPT_DIR/results/bench_results.csv"

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

OPTIONS:
  --tags TAG[,TAG,...]   Git tags to benchmark (required unless benchmarking HEAD)
  --M N                  Number of independent datasets      (default: 10)
  --K N                  MCMC chains per dataset             (default: 5)
  --data_seed N          Base seed for data generation       (default: 1234)
  --model_seed N         Base seed for sampler               (default: 4321)
  --n  N[,N,...]         Training-set sizes                  (default: 50000)
  --p  P[,P,...]         Feature counts                      (default: 50)
  --mode MODE            bart | xbart                        (default: bart)
  --trees N              Trees in ensemble                   (default: 200)
  --burnin N             Burn-in iters        (default: 200 bart / 15 xbart)
  --samples N            Posterior samples    (default: 1000 bart / 25 xbart)
  --threads N            Thread count (xbart)                (default: 1)
  --timeout SECS         Kill a run after N seconds; 0 = none (default: 0)
  --results FILE         CSV path (default: bench/results/bench_results.csv)
  --no-skip              Re-run even if row already in CSV
  --force-rebuild        Rebuild all tag binaries even if cached build exists
  --quick                n=2000, p=10, trees=50, burnin=50, samples=100, M=2, K=2
  -h, --help             Show this help
EOF
  exit 0
}

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --tags)        IFS=',' read -ra TAGS   <<< "$2"; shift 2 ;;
    --M)           M="$2";               shift 2 ;;
    --K)           K="$2";               shift 2 ;;
    --data_seed)   BASE_DATA_SEED="$2";  shift 2 ;;
    --model_seed)  BASE_MODEL_SEED="$2"; shift 2 ;;
    --n)           IFS=',' read -ra N_VALS <<< "$2"; shift 2 ;;
    --p)           IFS=',' read -ra P_VALS <<< "$2"; shift 2 ;;
    --mode)        MODE="$2";            shift 2 ;;
    --trees)       TREES="$2";           shift 2 ;;
    --burnin)      BURNIN="$2";          shift 2 ;;
    --samples)     SAMPLES="$2";         shift 2 ;;
    --threads)     THREADS="$2";         shift 2 ;;
    --timeout)     TIMEOUT_S="$2";       shift 2 ;;
    --results)     RESULTS_CSV="$2";     shift 2 ;;
    --no-skip)     SKIP_EXISTING=false;  shift ;;
    --force-rebuild) FORCE_REBUILD=true; shift ;;
    --quick)
      N_VALS=(2000); P_VALS=(10); TREES=50; BURNIN=50; SAMPLES=100; M=2; K=2
      shift ;;
    -h|--help) usage ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$BURNIN" ]];  then BURNIN=$([[ "$MODE" == "xbart" ]]  && echo 15   || echo 200);  fi
if [[ -z "$SAMPLES" ]]; then SAMPLES=$([[ "$MODE" == "xbart" ]] && echo 25   || echo 1000); fi

# ── Helpers ───────────────────────────────────────────────────────────────────
ncpus() { sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4; }

with_timeout() {
  if [[ "$TIMEOUT_S" -gt 0 ]]; then
    if   command -v gtimeout &>/dev/null; then gtimeout "$TIMEOUT_S" "$@"
    elif command -v timeout  &>/dev/null; then timeout  "$TIMEOUT_S" "$@"
    else
      echo "WARNING: --timeout set but neither gtimeout nor timeout found; ignoring." >&2
      "$@"
    fi
  else
    "$@"
  fi
}

build_for_tag() {
  local src="$1" bld="$2"
  local exe="$bld/faststochtree_bench"
  if [[ -f "$exe" ]] && ! $FORCE_REBUILD; then printf "  (using cached build)\n"; return; fi
  rm -rf "$bld"
  cmake -S "$src" -B "$bld" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_COLOR_DIAGNOSTICS=OFF \
    -DFETCHCONTENT_BASE_DIR="$FETCH_CACHE" \
    --log-level=ERROR > /dev/null 2>&1
  cmake --build "$bld" --target faststochtree_bench \
    -j"$(ncpus)" > /dev/null 2>&1
}

# Returns 0 (true) if (tag, mode, n, p, dataset_id, chain_id) already in CSV.
row_exists() {
  local tag="$1" n="$2" p="$3" dataset_id="$4" chain_id="$5"
  [[ -f "$RESULTS_CSV" ]] || return 1
  awk -F',' -v t="$tag" -v mo="$MODE" -v n="$n" -v p="$p" \
             -v d="$dataset_id" -v c="$chain_id" \
    'NR>1 && $1==t && $3==mo && $4==n && $5==p && $10==d && $11==c { found=1; exit }
     END { exit !found }' \
    "$RESULTS_CSV"
}

run_one() {
  local tag="$1" bench_exe="$2" n="$3" p="$4"

  printf "  RUN   %-26s  n=%-8s  p=%-4s  M=%d  K=%d\n" "$tag" "$n" "$p" "$M" "$K"

  for m in $(seq 0 $(( M - 1 ))); do
    local data_seed=$(( BASE_DATA_SEED + m ))

    for k in $(seq 0 $(( K - 1 ))); do
      local model_seed=$(( BASE_MODEL_SEED + m * K + k ))

      if $SKIP_EXISTING && row_exists "$tag" "$n" "$p" "$m" "$k"; then
        printf "    dataset %d/%d  chain %d/%d  SKIP\n" \
          $(( m + 1 )) "$M" $(( k + 1 )) "$K"
        continue
      fi

      local raw exit_code=0
      if [[ "$MODE" == "xbart" ]]; then
        raw=$(with_timeout "$bench_exe" \
          --mode       "$MODE"        \
          --n_train    "$n"           \
          --p          "$p"           \
          --trees      "$TREES"       \
          --burnin     "$BURNIN"      \
          --samples    "$SAMPLES"     \
          --threads    "$THREADS"     \
          --iters      1              \
          --seed       "$data_seed"   \
          --model_seed "$model_seed"  \
          --csv        /dev/null      \
          2>&1) || exit_code=$?
      else
        raw=$(with_timeout "$bench_exe" \
          --mode       "$MODE"        \
          --n_train    "$n"           \
          --p          "$p"           \
          --trees      "$TREES"       \
          --burnin     "$BURNIN"      \
          --samples    "$SAMPLES"     \
          --iters      1              \
          --seed       "$data_seed"   \
          --model_seed "$model_seed"  \
          --csv        /dev/null      \
          2>&1) || exit_code=$?
      fi

      if [[ $exit_code -ne 0 ]]; then
        printf "    dataset %d/%d  chain %d/%d  TIMEOUT/ERROR (exit %d)\n" \
          $(( m + 1 )) "$M" $(( k + 1 )) "$K" "$exit_code"
        continue
      fi

      local time_ms rmse sigma_post
      time_ms=$(   echo "$raw" | awk '/done in/         { print $(NF-1); exit }')
      rmse=$(      echo "$raw" | awk '/Test RMSE:/       { print $3;     exit }')
      sigma_post=$(echo "$raw" | awk '/Posterior sigma:/ { print $3;     exit }')

      local ts
      ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,%d,%d,%d,%s,%s,%s\n" \
        "$tag" "$ts" "$MODE" "$n" "$p" \
        "$TREES" "$BURNIN" "$SAMPLES" "$THREADS" \
        "$m" "$k" "$M" "$K" \
        "$time_ms" "$rmse" "$sigma_post" \
        >> "$RESULTS_CSV"

      printf "    dataset %d/%d  chain %d/%d  %s ms  RMSE=%s\n" \
        $(( m + 1 )) "$M" $(( k + 1 )) "$K" "$time_ms" "$rmse"
    done
  done
}

run_tag() {
  local tag="$1" bench_exe="$2"
  for n in "${N_VALS[@]}"; do
    for p in "${P_VALS[@]}"; do
      run_one "$tag" "$bench_exe" "$n" "$p"
    done
  done
}

# ── Init CSV ──────────────────────────────────────────────────────────────────
mkdir -p "$(dirname "$RESULTS_CSV")"
if [[ ! -f "$RESULTS_CSV" ]]; then
  echo "tag,timestamp,mode,n_train,p,trees,burnin,samples,threads,dataset_id,chain_id,M,K,time_ms,rmse,sigma_post" \
    > "$RESULTS_CSV"
fi

# ── Main ──────────────────────────────────────────────────────────────────────
if [[ ${#TAGS[@]} -eq 0 ]]; then
  current_tag=$(git -C "$REPO_DIR" describe --tags --exact-match 2>/dev/null \
                || git -C "$REPO_DIR" rev-parse --short HEAD)
  echo "=== Benchmarking current checkout: $current_tag ==="
  echo "    n=[${N_VALS[*]}]  p=[${P_VALS[*]}]  mode=$MODE"
  echo "    M=$M  K=$K  burnin=$BURNIN  samples=$SAMPLES"
  echo ""

  printf "  Building..."
  build_for_tag "$REPO_DIR" "$REPO_DIR/build-release"
  printf " done\n\n"
  run_tag "$current_tag" "$REPO_DIR/build-release/faststochtree_bench"

else
  echo "=== Tags to benchmark: ${TAGS[*]} ==="
  echo "    n=[${N_VALS[*]}]  p=[${P_VALS[*]}]  mode=$MODE"
  echo "    M=$M  K=$K  burnin=$BURNIN  samples=$SAMPLES"

  WT_BASE=$(mktemp -d)
  cleanup() {
    rm -rf "$WT_BASE"
    git -C "$REPO_DIR" worktree prune --quiet 2>/dev/null || true
  }
  trap cleanup EXIT

  mkdir -p "$BENCH_BUILDS" "$FETCH_CACHE"

  for tag in "${TAGS[@]}"; do
    echo ""
    echo "--- $tag ---"
    local_bld="$BENCH_BUILDS/$tag"
    local_exe="$local_bld/faststochtree_bench"

    if [[ ! -f "$local_exe" ]] || $FORCE_REBUILD; then
      wt="$WT_BASE/$tag"
      git -C "$REPO_DIR" worktree add --quiet "$wt" "refs/tags/$tag"
      printf "  Building..."
      build_for_tag "$wt" "$local_bld"
      printf " done\n"
      git -C "$REPO_DIR" worktree remove --force "$wt"
    else
      printf "  (using cached build)\n"
    fi

    run_tag "$tag" "$local_exe"
  done
fi

echo ""
echo "Results: $RESULTS_CSV"
