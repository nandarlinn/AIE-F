#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import subprocess
from pathlib import Path


DEFAULT_DOMAINS = [
    ("Social", "DomainTest/word_based_200/FB Comment.csv"),
    ("Law", "DomainTest/word_based_200/Law .csv"),
    ("News", "DomainTest/word_based_200/news1.csv"),
]


def read_text_lines(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        return []

    header = rows[0]
    text_idx = 0
    for i, name in enumerate(header):
        if name.strip().lower() in {"text", "text-mm", "sentence", "question", "answer", "content", "body"}:
            text_idx = i
            data_rows = rows[1:]
            break
    else:
        data_rows = rows

    lines = []
    for row in data_rows:
        if text_idx < len(row):
            text = " ".join(row[text_idx].split())
            if text:
                lines.append(text)
    return lines


def run_query(query_bin: Path, model: Path, lines: list[str]) -> str:
    proc = subprocess.run(
        [str(query_bin), str(model)],
        input="\n".join(lines) + "\n",
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout + "\n" + proc.stderr


def parse_query_output(output: str) -> tuple[float, int, int]:
    total_log10 = 0.0
    total_tokens = 0
    total_oov = 0

    total_re = re.compile(r"Total:\s+(-?\d+(?:\.\d+)?)\s+OOV:\s+(\d+)")
    tokens_re = re.compile(r"Tokens:\s+(\d+)")

    pending_total = None
    pending_oov = None
    for line in output.splitlines():
        total_match = total_re.search(line)
        if total_match:
            pending_total = float(total_match.group(1))
            pending_oov = int(total_match.group(2))
            continue

        tokens_match = tokens_re.search(line)
        if tokens_match and pending_total is not None:
            total_log10 += pending_total
            total_oov += pending_oov or 0
            total_tokens += int(tokens_match.group(1))
            pending_total = None
            pending_oov = None

    if total_tokens == 0:
        raise RuntimeError("Could not parse KenLM query output.")

    return total_log10, total_tokens, total_oov


def compute_ppl(total_log10: float, total_tokens: int) -> float:
    return math.pow(10.0, -total_log10 / total_tokens)


def write_chart(rows: list[dict], chart_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chart_path.parent.mkdir(parents=True, exist_ok=True)
    domains = [row["domain"] for row in rows]
    ppls = [row["ppl"] for row in rows]
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(domains, ppls, color=colors[: len(domains)], width=0.58)
    ax.set_title("Domain Perplexity with KenLM Word 5-gram")
    ax.set_ylabel("Perplexity (lower is easier)")
    ax.set_xlabel("Domain")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    ymax = max(ppls) if ppls else 1.0
    ax.set_ylim(0, ymax * 1.18)
    for bar, value in zip(bars, ppls):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    hardest = max(rows, key=lambda row: row["ppl"])
    ax.text(
        0.99,
        0.95,
        f"Hardest: {hardest['domain']}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "boxstyle": "round,pad=0.25"},
    )

    fig.tight_layout()
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute DomainTest perplexity with KenLM query.")
    parser.add_argument("--model", default="language_model_process/models/binary/word_5gram.binary")
    parser.add_argument("--query", default="/home/phantom/mosesdecoder/bin/query")
    parser.add_argument("--report", default="DomainTest/perplexity/reports/domain_ppl.tsv")
    parser.add_argument("--chart", default="DomainTest/perplexity/charts/domain_ppl_bar.png")
    args = parser.parse_args()

    model = Path(args.model)
    query_bin = Path(args.query)
    if not model.is_file():
        raise FileNotFoundError(f"Model not found: {model}")
    if not query_bin.is_file():
        raise FileNotFoundError(f"KenLM query not found: {query_bin}")

    rows = []
    for domain, path in DEFAULT_DOMAINS:
        csv_path = Path(path)
        lines = read_text_lines(csv_path)
        output = run_query(query_bin, model, lines)
        total_log10, tokens, oov = parse_query_output(output)
        rows.append(
            {
                "domain": domain,
                "file": str(csv_path),
                "sentences": len(lines),
                "tokens": tokens,
                "oov": oov,
                "log10_total": total_log10,
                "ppl": compute_ppl(total_log10, tokens),
            }
        )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["domain", "file", "sentences", "tokens", "oov", "log10_total", "ppl"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-domain-ppl")
    write_chart(rows, Path(args.chart))

    by_domain = {row["domain"]: row["ppl"] for row in rows}
    print(f"News PPL = {by_domain['News']:.4f}")
    print(f"Social PPL = {by_domain['Social']:.4f}")
    print(f"Law PPL = {by_domain['Law']:.4f}")
    print(f"Bar chart = {args.chart}")


if __name__ == "__main__":
    main()
