#!/usr/bin/env bash
# bench/run_suite.sh — faststochtree benchmark suite
#
# MODES
#   No --tags:   benchmark the current checkout (build in place, use existing
#                build-release/ dir if already up-to-date via cmake).
#   --tags ...:  for each tag, spin up a git worktree, build there, run, then
#                tear it down.  Your working tree is never touched.
#
# RESULTS
#   Appended to bench/results/bench_results.csv (or --results FILE).
#   Runs that already appear in the CSV are skipped (--no-skip to override).
#
# EARLY TAGS WARNING
#   v1–v4 can take thousands of seconds at n=50000.  Use --timeout or
#   restrict --n to small values when sweeping old tags.
#
# EXAMPLES
#   ./bench/run_suite.sh                              # current checkout, default grid
#   ./bench/run_suite.sh --tags v12-leaf-counts,v13-flat-obs
#   ./bench/run_suite.sh --tags v1-naive-cpp --n 5000 --timeout 120
#   ./bench/run_suite.sh --quick                      # tiny grid, test the script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Persistent per-tag build cache — survives across script invocations.
BENCH_BUILDS="$REPO_DIR/.bench-builds"
# Shared FetchContent download cache — GoogleTest is fetched once, ever.
FETCH_CACHE="$REPO_DIR/.cmake-fetch-cache"

# ── Defaults ─────────────────────────────────────────────────────────────────
N_VALS=(10000 50000)
P_VALS=(10 50)
TREES=200
BURNIN=""          # filled in after MODE is known
SAMPLES=""         # filled in after MODE is known
MODE="bart"
THREADS=1
TIMEOUT_S=0        # 0 = no timeout
SKIP_EXISTING=true
TAGS=()
RESULTS_CSV="$SCRIPT_DIR/results/bench_results.csv"

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

OPTIONS:
  --tags TAG[,TAG,...]   Git tags to benchmark via worktrees.
                         Omit to benchmark the current checkout.
  --n  N[,N,...]         Training-set sizes   (default: 10000,50000)
  --p  P[,P,...]         Feature counts       (default: 10,50)
  --mode MODE            bart | xbart         (default: bart)
  --trees N              Trees in ensemble    (default: 200)
  --burnin N             Burn-in iters        (default: 200 bart / 15 xbart)
  --samples N            Posterior samples    (default: 1000 bart / 25 xbart)
  --threads N            Thread count (xbart) (default: 1)
  --timeout SECS         Kill a run after N seconds; 0 = none (default: 0)
  --results FILE         CSV path (default: bench/results/bench_results.csv)
  --no-skip              Re-run even if result already recorded
  --quick                n=2000, p=10, trees=50, burnin=50, samples=100
  -h, --help             Show this help
EOF
  exit 0
}

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --tags)    IFS=',' read -ra TAGS   <<< "$2"; shift 2 ;;
    --n)       IFS=',' read -ra N_VALS <<< "$2"; shift 2 ;;
    --p)       IFS=',' read -ra P_VALS <<< "$2"; shift 2 ;;
    --mode)    MODE="$2";    shift 2 ;;
    --trees)   TREES="$2";   shift 2 ;;
    --burnin)  BURNIN="$2";  shift 2 ;;
    --samples) SAMPLES="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --timeout) TIMEOUT_S="$2"; shift 2 ;;
    --results) RESULTS_CSV="$2"; shift 2 ;;
    --no-skip) SKIP_EXISTING=false; shift ;;
    --quick)
      N_VALS=(2000); P_VALS=(10); TREES=50; BURNIN=50; SAMPLES=100; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Fill mode-specific defaults if not explicitly set
if [[ -z "$BURNIN" ]];  then BURNIN=$([[ "$MODE" == "xbart" ]]  && echo 15   || echo 200);  fi
if [[ -z "$SAMPLES" ]]; then SAMPLES=$([[ "$MODE" == "xbart" ]] && echo 25   || echo 1000); fi

# ── Helpers ───────────────────────────────────────────────────────────────────
ncpus() { sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4; }

# Run a command with optional timeout (tries gtimeout → timeout → bare).
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

# build_for_tag SRC BLD
#   Builds faststochtree_bench from source tree SRC into build dir BLD.
#   Skips the build entirely if the binary already exists (cached from a prior run).
#   All worktree builds share FETCH_CACHE so GoogleTest is only downloaded once.
build_for_tag() {
  local src="$1" bld="$2"
  local exe="$bld/faststochtree_bench"

  if [[ -f "$exe" ]]; then
    printf "  (using cached build)\n"
    return
  fi

  cmake -S "$src" -B "$bld" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_COLOR_DIAGNOSTICS=OFF \
    -DFETCHCONTENT_BASE_DIR="$FETCH_CACHE" \
    --log-level=ERROR \
    > /dev/null 2>&1
  cmake --build "$bld" --target faststochtree_bench \
    -j"$(ncpus)" \
    > /dev/null 2>&1
}

already_done() {
  local tag="$1" n="$2" p="$3"
  [[ -f "$RESULTS_CSV" ]] && \
    awk -F',' -v t="$tag" -v m="$MODE" -v n="$n" -v p="$p" \
      'NR>1 && $1==t && $3==m && $4==n && $5==p { found=1; exit }
       END { exit !found }' \
      "$RESULTS_CSV"
}

run_one() {
  local tag="$1" bench_exe="$2" n="$3" p="$4"

  if $SKIP_EXISTING && already_done "$tag" "$n" "$p"; then
    printf "  SKIP  %-26s  n=%-8s  p=%-4s  (already recorded)\n" "$tag" "$n" "$p"
    return
  fi

  printf "  RUN   %-26s  n=%-8s  p=%-4s  ..." "$tag" "$n" "$p"

  local raw exit_code=0
  if [[ "$MODE" == "xbart" ]]; then
    raw=$(with_timeout "$bench_exe" \
      --mode    "$MODE"    \
      --n_train "$n"       \
      --p       "$p"       \
      --trees   "$TREES"   \
      --burnin  "$BURNIN"  \
      --samples "$SAMPLES" \
      --threads "$THREADS" 2>&1) || exit_code=$?
  else
    raw=$(with_timeout "$bench_exe" \
      --mode    "$MODE"    \
      --n_train "$n"       \
      --p       "$p"       \
      --trees   "$TREES"   \
      --burnin  "$BURNIN"  \
      --samples "$SAMPLES" 2>&1) || exit_code=$?
  fi

  if [[ $exit_code -ne 0 ]]; then
    printf "  TIMEOUT/ERROR (exit %d)\n" "$exit_code"
    return
  fi

  local time_ms rmse sigma_post
  time_ms=$(   echo "$raw" | awk '/done in/          { print $(NF-1); exit }')
  rmse=$(      echo "$raw" | awk '/Test RMSE:/        { print $3;     exit }')
  sigma_post=$(echo "$raw" | awk '/Posterior sigma:/  { print $3;     exit }')

  local ts
  ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$tag" "$ts" "$MODE" "$n" "$p" \
    "$TREES" "$BURNIN" "$SAMPLES" "$THREADS" \
    "$time_ms" "$rmse" "$sigma_post" \
    >> "$RESULTS_CSV"

  printf "  %s ms   RMSE=%.4f\n" "$time_ms" "$rmse"
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
  echo "tag,timestamp,mode,n_train,p,trees,burnin,samples,threads,time_ms,rmse,sigma_post" \
    > "$RESULTS_CSV"
fi

# ── Main ──────────────────────────────────────────────────────────────────────
if [[ ${#TAGS[@]} -eq 0 ]]; then
  # ── Single-checkout mode ──
  current_tag=$(git -C "$REPO_DIR" describe --tags --exact-match 2>/dev/null \
                || git -C "$REPO_DIR" rev-parse --short HEAD)
  echo "=== Benchmarking current checkout: $current_tag ==="
  echo "    n=[${N_VALS[*]}]  p=[${P_VALS[*]}]  mode=$MODE  burnin=$BURNIN  samples=$SAMPLES"
  echo ""

  printf "  Building..."
  build_for_tag "$REPO_DIR" "$REPO_DIR/build-release"
  printf " done\n\n"
  run_tag "$current_tag" "$REPO_DIR/build-release/faststochtree_bench"

else
  # ── Multi-tag worktree mode ──
  echo "=== Tags to benchmark: ${TAGS[*]} ==="
  echo "    n=[${N_VALS[*]}]  p=[${P_VALS[*]}]  mode=$MODE  burnin=$BURNIN  samples=$SAMPLES"

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
      # Need to build: spin up a worktree, compile, then discard the source tree.
      # The compiled binary stays in .bench-builds/<tag>/ for future runs.
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
