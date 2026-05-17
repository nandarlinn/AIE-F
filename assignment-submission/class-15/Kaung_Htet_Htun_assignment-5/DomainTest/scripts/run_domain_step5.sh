#!/usr/bin/env bash
set -euo pipefail

perl DomainTest/scripts/domain_prepare_200_tokens.pl

echo "DomainTest Step 5 finished."
echo "Summary: DomainTest/reports/domain_200_summary.tsv"
