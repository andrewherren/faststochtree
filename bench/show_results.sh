#!/usr/bin/env bash
# bench/show_results.sh — pretty-print bench_results.csv or bench_results_mc.csv
#
# Auto-detects format: if the CSV has an "iter" column (MC format), results are
# aggregated (averaged) across iterations before display.
#
# Usage:
#   ./bench/show_results.sh                         # default: bench_results.csv
#   ./bench/show_results.sh --mc                    # default: bench_results_mc.csv
#   ./bench/show_results.sh [FILE]                  # explicit path
#   ./bench/show_results.sh --mode xbart
#   ./bench/show_results.sh --n 50000
#   ./bench/show_results.sh --tag v13

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV=""
FILTER_MODE=""
FILTER_N=""
FILTER_TAG=""
USE_MC=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --mc)   USE_MC=true; shift ;;
    --mode) FILTER_MODE="$2"; shift 2 ;;
    --n)    FILTER_N="$2";    shift 2 ;;
    --tag)  FILTER_TAG="$2";  shift 2 ;;
    -*)     echo "Unknown flag: $1" >&2; exit 1 ;;
    *)      CSV="$1"; shift ;;
  esac
done

# Resolve default CSV path
if [[ -z "$CSV" ]]; then
  if $USE_MC; then
    CSV="$SCRIPT_DIR/results/bench_results_mc.csv"
  else
    CSV="$SCRIPT_DIR/results/bench_results.csv"
  fi
fi

if [[ ! -f "$CSV" ]]; then
  echo "No results file found: $CSV"
  echo "Run ./bench/run_suite.sh or ./bench/run_suite_mc.sh first."
  exit 1
fi

# Auto-detect format by checking whether the header contains an "iter" column
header=$(head -1 "$CSV")
if echo "$header" | grep -q ',iter,'; then
  IS_MC=true
else
  IS_MC=false
fi

# ── MC format: aggregate across iterations ────────────────────────────────────
# Columns: tag,timestamp,mode,n_train,p,trees,burnin,samples,threads,iter,n_iters,time_ms,rmse,sigma_post
#          1    2         3     4      5  6      7       8        9     10  11      12      13    14

if $IS_MC; then
  printf "%-26s  %-6s  %-7s  %-4s  %6s  %9s  %8s\n" \
    "tag" "mode" "n_train" "p" "iters" "avg_ms" "avg_rmse"
  printf "%-26s  %-6s  %-7s  %-4s  %6s  %9s  %8s\n" \
    "--------------------------" "------" "-------" "----" \
    "------" "---------" "--------"

  awk -F',' -v fm="$FILTER_MODE" -v fn="$FILTER_N" -v ft="$FILTER_TAG" '
    NR == 1 { next }
    (fm == "" || $3 == fm) &&
    (fn == "" || $4 == fn) &&
    (ft == "" || index($1, ft) > 0) {
      key = $1 SUBSEP $3 SUBSEP $4 SUBSEP $5
      sum_ms[key]   += $12
      sum_rmse[key] += $13
      cnt[key]++
      tag_f[key]  = $1
      mode_f[key] = $3
      n_f[key]    = $4
      p_f[key]    = $5
    }
    END {
      for (k in cnt)
        printf "%s,%s,%s,%s,%d,%.0f,%.4f\n",
          tag_f[k], mode_f[k], n_f[k], p_f[k],
          cnt[k], sum_ms[k]/cnt[k], sum_rmse[k]/cnt[k]
    }
  ' "$CSV" \
    | sort -t',' -k2,2 -k3,3n -k4,4n -k1,1 \
    | awk -F',' '{
        printf "%-26s  %-6s  %-7s  %-4s  %6s  %9s  %8s\n",
               $1, $2, $3, $4, $5, $6, $7
      }'

# ── Single-run format ─────────────────────────────────────────────────────────
# Columns: tag,timestamp,mode,n_train,p,trees,burnin,samples,threads,time_ms,rmse,sigma_post
#          1    2         3     4      5  6      7       8        9     10     11    12
else
  printf "%-26s  %-6s  %-7s  %-4s  %9s  %8s  %9s\n" \
    "tag" "mode" "n_train" "p" "time_ms" "rmse" "sigma_post"
  printf "%-26s  %-6s  %-7s  %-4s  %9s  %8s  %9s\n" \
    "--------------------------" "------" "-------" "----" \
    "---------" "--------" "---------"

  awk -F',' -v fm="$FILTER_MODE" -v fn="$FILTER_N" -v ft="$FILTER_TAG" '
    NR == 1 { next }
    (fm == "" || $3 == fm) &&
    (fn == "" || $4 == fn) &&
    (ft == "" || index($1, ft) > 0) { print $0 }
  ' "$CSV" \
    | sort -t',' -k3,3 -k4,4n -k5,5n -k1,1 \
    | awk -F',' '{
        printf "%-26s  %-6s  %-7s  %-4s  %9s  %8s  %9s\n",
               $1, $3, $4, $5, $10, $11, $12
      }'
fi
