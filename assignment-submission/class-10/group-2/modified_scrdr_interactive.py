#!/usr/bin/env python3

## Movie-oriented SCRDR: coarse label in `--target` (e.g. Recommend / Not), not movie titles.
## Rules learn when predicted label != actual. Genre column uses op `has` (subset match, order-free).
## After a run, titles whose predicted conclusion is in `--in_set` are listed as matched recommendations.

import argparse
import ast
import json
import os
import re

import pandas as pd
from sklearn.metrics import classification_report


def _norm_genre_token(s: str) -> str:
    return str(s).strip().strip("'\"").lower()


def parse_genres_cell(cell) -> set[str]:
    """Turn a cell (list-like string or comma text) into a normalized genre set."""
    if pd.isna(cell):
        return set()
    s = str(cell).strip()
    if not s:
        return set()
    if s.startswith("[") or s.startswith("("):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple, set)):
                return {_norm_genre_token(x) for x in parsed if str(x).strip()}
        except (SyntaxError, ValueError, TypeError):
            pass
    parts = re.split(r"[,;]", s)
    return {_norm_genre_token(p) for p in parts if _norm_genre_token(p)}


def parse_rule_genre_tokens(val: str) -> set[str]:
    """User rule value: one genre or comma-separated; order ignored."""
    parts = re.split(r"[,;]", str(val).strip())
    return {_norm_genre_token(p) for p in parts if _norm_genre_token(p)}


def parse_in_set_labels(s: str) -> set[str]:
    """Comma-separated conclusion strings that count as 'recommended' for title collection."""
    return {p.strip() for p in str(s).split(",") if p.strip()}


class RDRNode:
    def __init__(self, condition=None, conclusion=None):
        self.condition = condition  # {'col':.., 'op':.., 'val':..}
        self.conclusion = str(conclusion) if conclusion is not None else None
        self.if_true = None
        self.if_false = None

    def evaluate(self, row):
        if not self.condition or not self.condition["col"]:
            return True
        col, op, val = self.condition["col"], self.condition["op"], self.condition["val"]
        row_val = row[col]
        try:
            if op == "==":
                return str(row_val) == str(val)
            if op == "<":
                return float(row_val) < float(val)
            if op == ">":
                return float(row_val) > float(val)
        except Exception:
            return str(row_val) == str(val)
        return False

    def to_dict(self):
        return {
            "condition": self.condition,
            "conclusion": self.conclusion,
            "if_true": self.if_true.to_dict() if self.if_true else None,
            "if_false": self.if_false.to_dict() if self.if_false else None,
        }

    @staticmethod
    def from_dict(data):
        if not data:
            return None
        node = RDRNode(data["condition"], data["conclusion"])
        node.if_true = RDRNode.from_dict(data["if_true"])
        node.if_false = RDRNode.from_dict(data["if_false"])
        return node


class MovieRDRNode(RDRNode):
    """RDR node with genre-column semantics: op `has` means row genres contain all rule genres."""

    def __init__(self, condition=None, conclusion=None, genre_col="genres"):
        super().__init__(condition, conclusion)
        self.genre_col = genre_col

    def evaluate(self, row):
        if not self.condition or not self.condition["col"]:
            return True
        col, op, val = self.condition["col"], self.condition["op"], self.condition["val"]
        if col == self.genre_col:
            try:
                if op == "has":
                    req = parse_rule_genre_tokens(val)
                    if not req:
                        return False
                    row_genres = parse_genres_cell(row[col])
                    return req <= row_genres
                if op == "==":
                    return str(row[col]) == str(val)
            except Exception:
                return False
            return False
        return super().evaluate(row)

    @staticmethod
    def from_dict(data, genre_col="genres"):
        if not data:
            return None
        node = MovieRDRNode(data["condition"], data["conclusion"], genre_col=genre_col)
        node.if_true = MovieRDRNode.from_dict(data["if_true"], genre_col=genre_col)
        node.if_false = MovieRDRNode.from_dict(data["if_false"], genre_col=genre_col)
        return node


class SCRDR_Engine:
    def __init__(self, target, default_conclusion="Unknown"):
        self.target = target
        self.root = RDRNode(conclusion=default_conclusion)

    def classify(self, row):
        curr = self.root
        last_match = self.root
        while curr:
            if curr.evaluate(row):
                last_match = curr
                curr = curr.if_true
            else:
                curr = curr.if_false
        return last_match


class MovieSCRDR_Engine(SCRDR_Engine):
    def __init__(
        self,
        target,
        genre_col="genres",
        title_col="title",
        default_conclusion="Unknown",
    ):
        self.genre_col = genre_col
        self.target = target
        self.title_col = title_col
        self.root = MovieRDRNode(conclusion=default_conclusion, genre_col=genre_col)

    def add_rule(self, row, last_node):
        print("\n[KNOWLEDGE ACQUISITION]")
        print(
            f"System predicted: '{last_node.conclusion}' but actual label is: '{row[self.target]}'"
        )

        feature_cols = [c for c in row.index if c not in (self.target, self.title_col)]

        def get_validated_input(prompt, options=None, check_cols=False):
            while True:
                val = input(prompt).strip()
                if val.lower() == "exit":
                    return "EXIT_COMMAND"
                if not val:
                    print("Input cannot be empty.")
                    continue
                if options and val not in options:
                    print(f"Invalid input. Choose from: {options}")
                    continue
                if check_cols and val not in feature_cols:
                    print(f"Invalid column. Available: {feature_cols}")
                    continue
                return val

        col = get_validated_input(
            "Enter feature column name (not label target or title column; or 'exit'): ",
            check_cols=True,
        )
        if col == "EXIT_COMMAND":
            return False

        if col == self.genre_col:
            op = get_validated_input(
                "Enter operator for genres (`has` = row must contain all listed genres): ",
                options=["has"],
            )
            if op == "EXIT_COMMAND":
                return False
            val = get_validated_input(
                "Enter genre(s): one name, or comma-separated for multiple (order ignored): "
            )
            if val == "EXIT_COMMAND":
                return False
        else:
            op = get_validated_input("Enter operator (==, <, >): ", options=["==", "<", ">"])
            if op == "EXIT_COMMAND":
                return False
            val = get_validated_input("Enter threshold / compare value: ")
            if val == "EXIT_COMMAND":
                return False

        new_node = MovieRDRNode(
            condition={"col": col, "op": op, "val": val},
            conclusion=row[self.target],
            genre_col=self.genre_col,
        )

        if last_node.evaluate(row):
            if last_node.if_true is None:
                last_node.if_true = new_node
            else:
                curr = last_node.if_true
                while curr.if_false:
                    curr = curr.if_false
                curr.if_false = new_node
        else:
            curr = last_node
            while curr.if_false:
                curr = curr.if_false
            curr.if_false = new_node
        return True


def main():
    help_epilog = """
Coarse-label workflow (not movie-title-as-label):
1. CSV must include `--target` (e.g. recommend) with values like Recommend / Not — ground truth for
   whether that row should be recommended.
2. Build mode: when predicted label != actual label, add rules (genres use `has`; other cols use
   ==, <, >). New rule conclusion is the row's actual label.
3. After the pass: rows whose *predicted* conclusion is in `--in_set` are listed using `--title_col`.
4. Type 'exit' at any prompt to save the tree and stop.
    """
    parser = argparse.ArgumentParser(
        description="Interactive RDR for recommendations: coarse label + list matched titles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=help_epilog,
    )
    parser.add_argument("--input", required=True, help="Dataset CSV file")
    parser.add_argument(
        "--target",
        default="recommend",
        help="Coarse ground-truth label column (e.g. recommend with Recommend / Not)",
    )
    parser.add_argument(
        "--title_col",
        default="title",
        help="Column printed when prediction is in --in_set (usually movie title)",
    )
    parser.add_argument(
        "--in_set",
        default="Recommend",
        help="Comma-separated predicted conclusion values that count as recommended (title listed)",
    )
    parser.add_argument(
        "--genre_col",
        default="genres",
        help="Column whose cells are list-like genre strings; rules use op `has` for subset match",
    )
    parser.add_argument("--tree", default="modified_rdr_movies.json", help="Output JSON model file")
    parser.add_argument("--exclude", nargs="*", help="Columns to ignore (optional)")
    parser.add_argument(
        "--mode",
        choices=["build", "test"],
        default="build",
        help="build: interactive training | test: run model only",
    )

    args = parser.parse_args()
    in_set_labels = parse_in_set_labels(args.in_set)

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    if args.exclude:
        df = df.drop(columns=[c for c in args.exclude if c in df.columns])

    for req in (args.target, args.title_col):
        if req not in df.columns:
            print(f"Error: required column '{req}' not in CSV. Columns: {list(df.columns)}")
            return

    engine = MovieSCRDR_Engine(
        args.target,
        genre_col=args.genre_col,
        title_col=args.title_col,
    )

    if os.path.exists(args.tree):
        with open(args.tree, "r") as f:
            engine.root = MovieRDRNode.from_dict(json.load(f), genre_col=args.genre_col)
        print(f"Loaded existing model: {args.tree}")

    y_true, y_pred = [], []
    total_rows = len(df)
    stopped_early = False

    print(
        f"--- Running in {args.mode.upper()} mode "
        f"(target={args.target}, title_col={args.title_col}, genre_col={args.genre_col}, "
        f"in_set={sorted(in_set_labels)}) ---"
    )

    for idx, row in df.iterrows():
        pred_node = engine.classify(row)
        actual = str(row[args.target])
        predicted = str(pred_node.conclusion).strip()

        if args.mode == "build" and predicted != actual:
            print(f"\n[Row {idx + 1}/{total_rows}] mismatch!")
            print(row.to_dict())

            success = engine.add_rule(row, pred_node)
            if not success:
                print("\nExiting and saving model...")
                stopped_early = True
                break

            with open(args.tree, "w") as f:
                json.dump(engine.root.to_dict(), f, indent=2)

            predicted = str(engine.classify(row).conclusion).strip()

        y_true.append(actual)
        y_pred.append(predicted)

    matched_titles: list[str] = []
    for _, row in df.iterrows():
        pred = str(engine.classify(row).conclusion).strip()
        if pred in in_set_labels:
            matched_titles.append(str(row[args.title_col]))

    print("\n--- FINAL PERFORMANCE SUMMARY (coarse labels) ---")
    if stopped_early:
        print(f"(Partial results based on first {len(y_true)} rows)")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("\n--- MATCHED TITLES (predicted conclusion in in_set) ---")
    print(f"in_set: {sorted(in_set_labels)}")
    if stopped_early:
        print("( label stats above are partial; title list uses final tree on full dataset )")
    for t in matched_titles:
        print(t)
    print(f"[count] {len(matched_titles)}")


if __name__ == "__main__":
    main()