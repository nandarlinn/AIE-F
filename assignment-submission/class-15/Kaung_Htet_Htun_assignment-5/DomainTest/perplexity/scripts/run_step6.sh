#!/usr/bin/env bash
set -euo pipefail

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-domain-ppl}" \
python3 DomainTest/perplexity/scripts/compute_domain_ppl.py
