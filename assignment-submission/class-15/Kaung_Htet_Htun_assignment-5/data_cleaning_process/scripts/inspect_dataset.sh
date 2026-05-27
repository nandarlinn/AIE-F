#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:-Dataset}"
REPORT_DIR="${2:-data_cleaning_process/reports}"

mkdir -p "$REPORT_DIR"

{
  echo "== CSV files =="
  find "$INPUT_DIR" -maxdepth 1 -type f -iname '*.csv' | sort
  echo
  echo "== Line counts =="
  wc -l "$INPUT_DIR"/*.csv
  echo
  echo "== File encoding/type =="
  file "$INPUT_DIR"/*.csv
} > "$REPORT_DIR/dataset_overview.txt"

perl -CS -Mopen=:std,:encoding\(UTF-8\) -ne '
  chomp;
  $total++;
  $dup{$_}++;
  while (/([^\p{Myanmar}\p{Space}\p{P}\p{S}\p{N}]+)/g) {
    $non_mm{$1}++;
  }
  END {
    $dups = 0;
    for $line (keys %dup) {
      $dups++ if $dup{$line} > 1;
    }
    print "total_lines\t$total\n";
    print "duplicate_exact_lines\t$dups\n";
    print "\nnon_myanmar_tokens_top_50\n";
    $i = 0;
    for $token (sort { $non_mm{$b} <=> $non_mm{$a} || $a cmp $b } keys %non_mm) {
      print "$non_mm{$token}\t$token\n";
      last if ++$i >= 50;
    }
  }
' "$INPUT_DIR"/*.csv > "$REPORT_DIR/dataset_noise_report.tsv"

echo "Reports written to $REPORT_DIR"
