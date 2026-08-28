#!/usr/bin/env python3
"""GeoDIP command line predictor.

Examples:
    python geodip.py predictor=LR range=5C input=9948_geno.csv output=out.csv
    python geodip.py -p ML -r EAS -a XGB -i 9948_geno.csv -o out.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

from core import (
    ALGORITHM_ALIASES,
    ALGORITHM_BY_PKL,
    InputError,
    load_genotype_table,
    render_terminal_table,
    run_lr,
    run_ml,
    write_result,
)


VERSION = "1.0.0"

OPTION_ALIASES = {
    "predictor": "predictor",
    "p": "predictor",
    "range": "range",
    "r": "range",
    "algorithm": "algorithm",
    "a": "algorithm",
    "input": "input",
    "i": "input",
    "output": "output",
    "o": "output",
}


class UsageError(ValueError):
    """Raised when the command line arguments are invalid."""


def print_help():
    print(
        """GeoDIP Command Line Predictor v"""
        + VERSION
        + """

Usage:
  python geodip.py predictor=LR|ML range=5C|EAS input=FILE output=FILE [algorithm=ALGO]
  python geodip.py -p LR|ML -r 5C|EAS -i FILE -o FILE [-a ALGO]

Required options:
  predictor, p   Prediction mode: LR or ML.
  range, r       Prediction range: 5C or EAS.
  input, i       Genotype input file (CSV or TSV). The first column must be the sample ID.
  output, o      Output result file.

ML options:
  algorithm, a   ML algorithm. Default is XGB for both 5C and EAS.
                 Choices: LR, GNB, KNN, SVM, RF, HGB, XGB, AGB, MLP.

The input path can also be given as a positional argument.
Run 'python geodip.py --version' to print the version.
"""
    )


def _set_option(options, key, value):
    if key in options:
        raise UsageError(f"Option '{key}' was specified more than once")
    options[key] = value


def _consume_value(argv, index, key):
    if index + 1 >= len(argv):
        raise UsageError(f"Option '{key}' requires a value")
    return argv[index + 1], index + 2


def parse_argv(argv):
    """Parse both key=value tokens and conventional flag tokens."""
    options = {}
    positionals = []
    index = 0
    while index < len(argv):
        token = argv[index]

        if token in ("-h", "--help", "help"):
            options["help"] = "1"
            index += 1
            continue
        if token in ("--version", "version"):
            options["version"] = "1"
            index += 1
            continue

        if token.startswith("--") and "=" in token:
            raw_key, value = token[2:].split("=", 1)
            key = OPTION_ALIASES.get(raw_key)
            if key is None:
                raise UsageError(f"Unknown option: --{raw_key}")
            _set_option(options, key, value)
            index += 1
        elif token.startswith("-") and not token.startswith("--") and "=" in token:
            raw_key, value = token[1:].split("=", 1)
            key = OPTION_ALIASES.get(raw_key)
            if key is None:
                raise UsageError(f"Unknown option: -{raw_key}")
            _set_option(options, key, value)
            index += 1
        elif token.startswith("--"):
            raw_key = token[2:]
            key = OPTION_ALIASES.get(raw_key)
            if key is None:
                raise UsageError(f"Unknown option: --{raw_key}")
            value, index = _consume_value(argv, index, key)
            _set_option(options, key, value)
        elif token.startswith("-") and len(token) == 2:
            raw_key = token[1:]
            key = OPTION_ALIASES.get(raw_key)
            if key is None:
                raise UsageError(f"Unknown option: -{raw_key}")
            value, index = _consume_value(argv, index, key)
            _set_option(options, key, value)
        elif token.startswith("-") and not token.startswith("--"):
            raw_key = token[1:]
            key = OPTION_ALIASES.get(raw_key)
            if key is None:
                raise UsageError(f"Unknown option: -{raw_key}")
            value, index = _consume_value(argv, index, key)
            _set_option(options, key, value)
        elif "=" in token:
            raw_key, value = token.split("=", 1)
            key = OPTION_ALIASES.get(raw_key)
            if key is not None:
                _set_option(options, key, value)
            else:
                positionals.append(token)
            index += 1
        else:
            positionals.append(token)
            index += 1

    if options.get("input") is None and positionals:
        options["input"] = positionals.pop(0)
    if positionals:
        raise UsageError(f"Unexpected positional argument(s): {positionals}")
    return options


def normalize_predictor(value):
    upper = str(value).strip().upper()
    if upper not in ("LR", "ML"):
        if not upper:
            raise UsageError("Missing required option: predictor")
        raise UsageError("predictor must be one of: LR, ML")
    return upper


def normalize_range(value):
    upper = str(value).strip().upper()
    if upper not in ("5C", "EAS"):
        if not upper:
            raise UsageError("Missing required option: range")
        raise UsageError("range must be one of: 5C, EAS")
    return upper


def normalize_algorithm(value):
    upper = str(value).strip().upper()
    if upper in ALGORITHM_ALIASES:
        return upper
    if str(value).strip().lower() in ALGORITHM_BY_PKL:
        return ALGORITHM_BY_PKL[str(value).strip().lower()]
    choices = ", ".join(sorted(ALGORITHM_ALIASES))
    raise UsageError(f"algorithm must be one of: {choices}")


def print_summary(summary):
    range_label = (
        "Continental populations"
        if summary["range"] == "5C"
        else "East Asians"
    )
    if summary["mode"] == "LR":
        print("Mode        : LR")
        print(f"Range       : {range_label}")
        print(f"Frequency   : {summary['freq_path']}")
        print(f"Samples     : {summary['sample_count']}")
    else:
        print("Mode        : ML")
        print(f"Range       : {range_label}")
        print(f"Algorithm   : {summary['algorithm']} ({summary['algorithm_key']})")
        print(f"Model       : {summary['model_path']}")
        print(f"Samples     : {summary['sample_count']}")
        print(f"Classes     : {', '.join(summary['classes'])}")


def run(argv=None):
    """Run the CLI and return an exit code."""
    if argv is None:
        argv = sys.argv[1:]

    try:
        options = parse_argv(argv)
        if options.get("help"):
            print_help()
            return 0
        if options.get("version"):
            print(f"geodip {VERSION}")
            return 0
        if not options:
            print_help()
            return 2

        predictor = normalize_predictor(options.get("predictor", ""))
        range_name = normalize_range(options.get("range", ""))
        input_path = options.get("input")
        if not input_path:
            raise UsageError("Missing required option: input")
        output_path = options.get("output")
        if not output_path:
            raise UsageError("Missing required option: output")
        raw = load_genotype_table(input_path)
        if predictor == "LR":
            if options.get("algorithm") is not None:
                raise UsageError("algorithm is only valid with predictor=ML")
            headers, rows, summary = run_lr(raw, range_name)
        else:
            algorithm_alias = normalize_algorithm(options.get("algorithm", "XGB"))
            headers, rows, summary = run_ml(raw, range_name, algorithm_alias)

        write_result(headers, rows, output_path)
        print()
        print_summary(summary)
        print()
        print(render_terminal_table(headers, rows))
        print()
        print(f"Results written to: {Path(output_path).resolve()}")
        return 0
    except (UsageError, InputError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "Run 'python geodip.py --help' for usage.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(run())
