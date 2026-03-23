#!/usr/bin/env bash
# bench/show_results.sh — pretty-print bench_results.csv in the terminal
#
# Usage: ./bench/show_results.sh [FILE]          (default: bench/results/bench_results.csv)
#        ./bench/show_results.sh --mode xbart    (filter by mode)
#        ./bench/show_results.sh --n 50000       (filter by n_train)
#        ./bench/show_results.sh --tag v13-flat-obs  (filter by tag substring)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV="$SCRIPT_DIR/results/bench_results.csv"
FILTER_MODE=""
FILTER_N=""
FILTER_TAG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --mode) FILTER_MODE="$2"; shift 2 ;;
    --n)    FILTER_N="$2";    shift 2 ;;
    --tag)  FILTER_TAG="$2";  shift 2 ;;
    -*)     echo "Unknown flag: $1" >&2; exit 1 ;;
    *)      CSV="$1"; shift ;;
  esac
done

if [[ ! -f "$CSV" ]]; then
  echo "No results file found: $CSV"
  echo "Run ./bench/run_suite.sh first."
  exit 1
fi

# Print a subset of columns in a readable order, sorted by mode → n → p → tag.
# Columns in CSV: tag,timestamp,mode,n_train,p,trees,burnin,samples,threads,time_ms,rmse,sigma_post
#                  1       2      3     4    5   6      7       8      9      10     11     12

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
