#!/usr/bin/env bash
# bench/run_suite_mc.sh — multi-iteration benchmark suite
#
# Runs each (tag, n, p) configuration N_ITERS times with independent random
# seeds and appends one CSV row per iteration to bench_results_mc.csv.
# The binary (bench/main.cpp) handles all CSV writing; this script handles
# builds, skip logic, and tag/worktree orchestration.
#
# Compared to run_suite.sh:
#   - Default grid is n=50000, p=50 only (slow versions are very expensive
#     to repeat; pass --n/--p to override)
#   - Adds --iters N (default 10)
#   - CSV has extra columns: iter, n_iters
#   - Skip logic counts existing rows and skips when count >= n_iters
#
# EXAMPLES
#   ./bench/run_suite_mc.sh
#   ./bench/run_suite_mc.sh --tags v13-flat-obs --iters 20
#   ./bench/run_suite_mc.sh --tags v1-naive-cpp --n 10000 --timeout 300
#   ./bench/run_suite_mc.sh --quick

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
N_ITERS=10
BASE_SEED=12345
TIMEOUT_S=0
SKIP_EXISTING=true
TAGS=()
RESULTS_CSV="$SCRIPT_DIR/results/bench_results_mc.csv"

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

OPTIONS:
  --tags TAG[,TAG,...]   Git tags to benchmark via worktrees.
                         Omit to benchmark the current checkout.
  --iters N              Independent iterations per configuration (default: 10)
  --seed N               Base random seed; iter i uses seed+i-1 (default: 12345)
  --n  N[,N,...]         Training-set sizes   (default: 50000)
  --p  P[,P,...]         Feature counts       (default: 50)
  --mode MODE            bart | xbart         (default: bart)
  --trees N              Trees in ensemble    (default: 200)
  --burnin N             Burn-in iters        (default: 200 bart / 15 xbart)
  --samples N            Posterior samples    (default: 1000 bart / 25 xbart)
  --threads N            Thread count (xbart) (default: 1)
  --timeout SECS         Kill a run after N seconds; 0 = none (default: 0)
  --results FILE         CSV path (default: bench/results/bench_results_mc.csv)
  --no-skip              Re-run even if all iterations already recorded
  --quick                n=2000, p=10, trees=50, burnin=50, samples=100, iters=3
  -h, --help             Show this help
EOF
  exit 0
}

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --tags)    IFS=',' read -ra TAGS   <<< "$2"; shift 2 ;;
    --iters)   N_ITERS="$2";    shift 2 ;;
    --seed)    BASE_SEED="$2"; shift 2 ;;
    --n)       IFS=',' read -ra N_VALS <<< "$2"; shift 2 ;;
    --p)       IFS=',' read -ra P_VALS <<< "$2"; shift 2 ;;
    --mode)    MODE="$2";     shift 2 ;;
    --trees)   TREES="$2";    shift 2 ;;
    --burnin)  BURNIN="$2";   shift 2 ;;
    --samples) SAMPLES="$2";  shift 2 ;;
    --threads) THREADS="$2";  shift 2 ;;
    --timeout) TIMEOUT_S="$2"; shift 2 ;;
    --results) RESULTS_CSV="$2"; shift 2 ;;
    --no-skip) SKIP_EXISTING=false; shift ;;
    --quick)
      N_VALS=(2000); P_VALS=(10); TREES=50; BURNIN=50; SAMPLES=100; N_ITERS=3
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
  if [[ -f "$exe" ]]; then printf "  (using cached build)\n"; return; fi
  cmake -S "$src" -B "$bld" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_COLOR_DIAGNOSTICS=OFF \
    -DFETCHCONTENT_BASE_DIR="$FETCH_CACHE" \
    --log-level=ERROR > /dev/null 2>&1
  cmake --build "$bld" --target faststochtree_bench \
    -j"$(ncpus)" > /dev/null 2>&1
}

# Count how many iterations are already recorded for (tag, mode, n, p).
# CSV columns: tag,timestamp,mode,n_train,p,...
count_existing() {
  local tag="$1" n="$2" p="$3"
  [[ -f "$RESULTS_CSV" ]] || { echo 0; return; }
  awk -F',' -v t="$tag" -v m="$MODE" -v n="$n" -v p="$p" \
    'NR>1 && $1==t && $3==m && $4==n && $5==p { count++ }
     END { print count+0 }' \
    "$RESULTS_CSV"
}

run_one() {
  local tag="$1" bench_exe="$2" n="$3" p="$4"

  local existing
  existing=$(count_existing "$tag" "$n" "$p")

  if $SKIP_EXISTING && [[ "$existing" -ge "$N_ITERS" ]]; then
    printf "  SKIP  %-26s  n=%-8s  p=%-4s  (already have %d/%d iterations)\n" \
      "$tag" "$n" "$p" "$existing" "$N_ITERS"
    return
  fi

  if [[ "$existing" -gt 0 ]]; then
    printf "  WARN  %-26s  n=%-8s  p=%-4s  (%d partial iterations in CSV; re-running all)\n" \
      "$tag" "$n" "$p" "$existing"
  else
    printf "  RUN   %-26s  n=%-8s  p=%-4s\n" "$tag" "$n" "$p"
  fi

  # Iterate in the shell, calling the binary once per iteration with a
  # varied --seed.  This is backward-compatible with old tag binaries that
  # predate --iters/--csv support: they accept --seed and emit the original
  # single-run output format which we parse here.
  for iter in $(seq 1 "$N_ITERS"); do
    local iter_seed=$(( BASE_SEED + iter - 1 ))

    local raw exit_code=0
    if [[ "$MODE" == "xbart" ]]; then
      raw=$(with_timeout "$bench_exe" \
        --mode    "$MODE"      \
        --n_train "$n"         \
        --p       "$p"         \
        --trees   "$TREES"     \
        --burnin  "$BURNIN"    \
        --samples "$SAMPLES"   \
        --threads "$THREADS"   \
        --iters   1            \
        --seed    "$iter_seed" \
        2>&1) || exit_code=$?
    else
      raw=$(with_timeout "$bench_exe" \
        --mode    "$MODE"      \
        --n_train "$n"         \
        --p       "$p"         \
        --trees   "$TREES"     \
        --burnin  "$BURNIN"    \
        --samples "$SAMPLES"   \
        --iters   1            \
        --seed    "$iter_seed" \
        2>&1) || exit_code=$?
    fi

    if [[ $exit_code -ne 0 ]]; then
      printf "    iter %d/%d  TIMEOUT/ERROR (exit %d)\n" "$iter" "$N_ITERS" "$exit_code"
      continue
    fi

    local time_ms rmse sigma_post
    time_ms=$(   echo "$raw" | awk '/done in/         { print $(NF-1); exit }')
    rmse=$(      echo "$raw" | awk '/Test RMSE:/       { print $3;     exit }')
    sigma_post=$(echo "$raw" | awk '/Posterior sigma:/ { print $3;     exit }')

    local ts
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,%d,%s,%s,%s\n" \
      "$tag" "$ts" "$MODE" "$n" "$p" \
      "$TREES" "$BURNIN" "$SAMPLES" "$THREADS" \
      "$iter" "$N_ITERS" \
      "$time_ms" "$rmse" "$sigma_post" \
      >> "$RESULTS_CSV"

    printf "    iter %d/%d  %s ms   RMSE=%s\n" "$iter" "$N_ITERS" "$time_ms" "$rmse"
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
  echo "tag,timestamp,mode,n_train,p,trees,burnin,samples,threads,iter,n_iters,time_ms,rmse,sigma_post" \
    > "$RESULTS_CSV"
fi

# ── Main ──────────────────────────────────────────────────────────────────────
if [[ ${#TAGS[@]} -eq 0 ]]; then
  current_tag=$(git -C "$REPO_DIR" describe --tags --exact-match 2>/dev/null \
                || git -C "$REPO_DIR" rev-parse --short HEAD)
  echo "=== Benchmarking current checkout: $current_tag ==="
  echo "    n=[${N_VALS[*]}]  p=[${P_VALS[*]}]  mode=$MODE  iters=$N_ITERS"
  echo "    burnin=$BURNIN  samples=$SAMPLES"
  echo ""

  printf "  Building..."
  build_for_tag "$REPO_DIR" "$REPO_DIR/build-release"
  printf " done\n\n"
  run_tag "$current_tag" "$REPO_DIR/build-release/faststochtree_bench"

else
  echo "=== Tags to benchmark: ${TAGS[*]} ==="
  echo "    n=[${N_VALS[*]}]  p=[${P_VALS[*]}]  mode=$MODE  iters=$N_ITERS"
  echo "    burnin=$BURNIN  samples=$SAMPLES"

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

    if [[ ! -f "$local_exe" ]]; then
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
