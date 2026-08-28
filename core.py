"""Core prediction logic shared by the GeoDIP command line tool."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# The bundled XGBoost models were pickled with a top-level module named
# ``model_classes``. Make our local copy importable before joblib.load() runs.
_CLIENT_DIR = Path(__file__).resolve().parent
if str(_CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CLIENT_DIR))
import model_classes  # noqa: F401


FREQ_DATA_DIR = _CLIENT_DIR / "data" / "freq"
MODEL_DATA_DIR = _CLIENT_DIR / "data" / "models"

MIN_FREQ = 1e-8

RANGE_CONFIG = {
    "5C": {
        "freq_file": FREQ_DATA_DIR / "freq_5c.csv",
        "populations": ["AFR", "AMR", "EAS", "EUR", "SAS"],
    },
    "EAS": {
        "freq_file": FREQ_DATA_DIR / "freq_eas.csv",
        "populations": ["HAN", "JPT", "SEAS"],
    },
}

ALGORITHM_ALIASES = {
    "LR": "logistic_regression",
    "GNB": "naive_bayes",
    "KNN": "knn",
    "SVM": "svm",
    "RF": "random_forest",
    "HGB": "gradient_boosting",
    "XGB": "xgboost",
    "AGB": "adaboost",
    "MLP": "mlp",
}

ALGORITHM_BY_PKL = {value: key for key, value in ALGORITHM_ALIASES.items()}

MISSING_TOKENS = {
    "",
    ".",
    "-",
    "?",
    ",",
    "NA",
    "N/A",
    "NAN",
    "NULL",
    "NONE",
    "MISSING",
}


class InputError(ValueError):
    """Raised when user-provided data cannot be processed."""


def detect_delimiter(path):
    """Pick the most likely delimiter for a genotype table."""
    suffix = Path(path).suffix.lower()
    if suffix in (".tsv", ".txt", ".tab"):
        return "\t"
    if suffix == ".csv":
        return ","
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            sample = fh.read(4096)
    except OSError:
        return ","
    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    if "\t" in first_line and "," not in first_line:
        return "\t"
    return ","


def load_genotype_table(path):
    """Load and validate a genotype table.

    The first column is treated as the sample ID column and every later column
    as one marker. All values are kept as strings so genotype validation can
    happen later.
    """
    input_path = Path(path)
    if not input_path.exists():
        raise InputError(f"Input file does not exist: {input_path}")
    delimiter = detect_delimiter(input_path)
    try:
        raw = pd.read_csv(
            input_path,
            dtype=str,
            sep=delimiter,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    except Exception as exc:
        raise InputError(
            f"Could not parse input file as {delimiter!r}-delimited text: {exc}"
        ) from exc

    if raw.shape[1] < 2:
        raise InputError(
            "Input file must contain a sample ID column and at least one marker column"
        )

    raw.columns = [str(col).strip() for col in raw.columns]
    if not raw.columns[0]:
        raise InputError("The first column must have a non-empty sample ID header")
    if raw.columns.duplicated().any():
        duplicates = sorted(
            {col for col in raw.columns[raw.columns.duplicated()]}
        )
        raise InputError(f"Duplicate marker columns are not allowed: {duplicates}")
    if raw.empty:
        raise InputError("Input file does not contain any sample rows")

    sample_col = raw.columns[0]
    sample_ids = raw[sample_col].astype(str).str.strip()
    empty_samples = raw.index[sample_ids.eq("")]
    if len(empty_samples):
        raise InputError(f"Sample ID is empty on row(s): {list(empty_samples + 2)}")
    if sample_ids.duplicated().any():
        duplicates = sample_ids[sample_ids.duplicated()].unique().tolist()
        raise InputError(f"Duplicate sample IDs are not allowed: {duplicates}")

    return raw


def _clean_value(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.upper() in MISSING_TOKENS:
        return None
    return text


def normalize_genotype(value):
    """Return a normalized genotype string, or None for a missing call."""
    text = _clean_value(value)
    if text is None:
        return None
    return text.replace("/", "|")


def _lr_pair(value):
    """Convert a normalized genotype to the pair notation used by LR scoring."""
    if value in ("0|0", "0|1", "1|0", "1|1"):
        return value
    if value in ("0", "1", "2"):
        return {"0": "0|0", "1": "0|1", "2": "1|1"}[value]
    return None


def _is_ml_genotype(value):
    if value in ("0|0", "0|1", "1|0", "1|1"):
        return True
    return value in ("0", "1", "2")


def validate_relevant_cells(raw, markers, mode):
    """Raise InputError when a marker used by the analysis has a bad call."""
    if not markers:
        return
    errors = []
    sample_col = raw.columns[0]
    for _, row in raw.iterrows():
        sample_id = str(row[sample_col]).strip()
        for marker in markers:
            if marker not in raw.columns:
                continue
            value = normalize_genotype(row[marker])
            if value is None:
                continue
            if mode == "LR" and _lr_pair(value) is None:
                errors.append(f"{sample_id} / {marker}: {row[marker]!r}")
            elif mode == "ML" and not _is_ml_genotype(value):
                errors.append(f"{sample_id} / {marker}: {row[marker]!r}")
            if len(errors) >= 20:
                raise InputError(
                    "Invalid genotype values found in analysis markers "
                    f"(showing first 20): {errors}"
                )
    if errors:
        raise InputError(
            "Invalid genotype values found in analysis markers "
            f"(showing first 20): {errors}"
        )


def load_freq_matrix(range_name):
    """Load the population allele-frequency table for one prediction range."""
    config = RANGE_CONFIG[range_name]
    path = config["freq_file"]
    if not path.exists():
        raise InputError(f"Frequency file does not exist: {path}")

    populations = config["populations"]
    freq_data = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = [col.strip().upper() for col in next(reader)]
        missing = [pop for pop in populations if pop not in header[1:]]
        if missing:
            raise InputError(
                f"Frequency file is missing required population columns: {missing}"
            )
        col_index = {col: idx for idx, col in enumerate(header)}
        for row in reader:
            if not row or not row[0].strip():
                continue
            site_id = row[0].strip()
            try:
                freq_data[site_id] = {
                    pop: float(row[col_index[pop]]) for pop in populations
                }
            except (IndexError, ValueError) as exc:
                raise InputError(
                    f"Invalid frequency value for site {site_id} in {path.name}"
                ) from exc
    if not freq_data:
        raise InputError(f"Frequency file contains no markers: {path}")
    return freq_data


def calculate_genotype_freq(genotype_pair, freq):
    """Compute genotype frequency with the same rules as the original LR.py."""
    if genotype_pair == "0|0":
        gf = (1 - freq) * (1 - freq)
    elif genotype_pair in ("0|1", "1|0"):
        gf = 2 * freq * (1 - freq)
    elif genotype_pair == "1|1":
        gf = freq * freq
    else:
        raise InputError(f"Unknown genotype for frequency calculation: {genotype_pair}")
    return max(gf, MIN_FREQ)


def run_lr(raw, range_name):
    """Run likelihood-ratio prediction and return (headers, rows, summary)."""
    config = RANGE_CONFIG[range_name]
    populations = config["populations"]
    freq_data = load_freq_matrix(range_name)

    input_markers = set(raw.columns[1:])
    common_sites = [site for site in freq_data if site in input_markers]
    missing_sites = [site for site in freq_data if site not in input_markers]
    if not common_sites:
        raise InputError(
            "No input markers overlap with the frequency table markers; "
            "check that the input file and prediction range match"
        )
    validate_relevant_cells(raw, common_sites, "LR")

    sample_col = raw.columns[0]
    headers = (
        ["sample_id"]
        + [f"PAMP_{pop}" for pop in populations]
        + ["LR", "predicted_pop", "valid_loci_count"]
    )
    rows = []
    for _, row in raw.iterrows():
        sample_id = str(row[sample_col]).strip()
        pamp_log = {pop: 0.0 for pop in populations}
        valid_sites = 0

        for site in common_sites:
            pair = _lr_pair(normalize_genotype(row[site]))
            if pair is None:
                continue
            for pop in populations:
                gf = calculate_genotype_freq(pair, freq_data[site][pop])
                pamp_log[pop] += math.log10(gf)
            valid_sites += 1

        pamp = {pop: 10 ** value if valid_sites else 0.0 for pop, value in pamp_log.items()}
        output_row = {
            "sample_id": sample_id,
            **{f"PAMP_{pop}": pamp[pop] for pop in populations},
            "valid_loci_count": valid_sites,
        }
        if valid_sites == 0:
            output_row["LR"] = None
            output_row["predicted_pop"] = "NA"
        else:
            ranked = sorted(populations, key=lambda pop: pamp[pop], reverse=True)
            best, second = ranked[0], ranked[1]
            lr = pamp[best] / pamp[second] if pamp[second] > 0 else float("inf")
            output_row["LR"] = lr
            output_row["predicted_pop"] = best
        rows.append([output_row[col] for col in headers])

    summary = {
        "mode": "LR",
        "range": range_name,
        "freq_path": config["freq_file"],
        "input_marker_count": len(input_markers),
        "freq_marker_count": len(freq_data),
        "common_site_count": len(common_sites),
        "missing_site_count": len(missing_sites),
        "sample_count": len(raw),
    }
    return headers, rows, summary


def encode_genotype(value):
    """Convert a diploid genotype to allele dosage for ML features."""
    text = normalize_genotype(value)
    if text is None:
        return np.nan
    if "|" in text:
        left, right = text.split("|")
        return int(left) + int(right)
    return float(text)


def load_ml_model(range_name, algorithm_key):
    """Load one tuned scikit-learn pipeline and its metadata."""
    pkl_path = MODEL_DATA_DIR / range_name / f"{algorithm_key}.pkl"
    metadata_path = MODEL_DATA_DIR / range_name / "model_metadata.json"
    if not pkl_path.exists():
        raise InputError(f"Model file does not exist: {pkl_path}")
    if not metadata_path.exists():
        raise InputError(f"Model metadata does not exist: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    try:
        model = joblib.load(pkl_path)
    except Exception as exc:
        raise InputError(f"Could not load model {pkl_path.name}: {exc}") from exc
    return model, metadata, pkl_path


def run_ml(raw, range_name, algorithm_alias):
    """Run ML prediction and return (headers, rows, summary)."""
    algorithm_key = ALGORITHM_ALIASES[algorithm_alias]
    model, metadata, model_path = load_ml_model(range_name, algorithm_key)
    feature_columns = list(metadata["feature_columns"])
    classes = [str(cls) for cls in model.classes_]

    input_markers = set(raw.columns[1:])
    missing_features = [col for col in feature_columns if col not in input_markers]
    present_features = [col for col in feature_columns if col in input_markers]
    if not present_features:
        raise InputError(
            "No input markers match the model feature columns; "
            "check that the input file and prediction range match"
        )
    validate_relevant_cells(raw, present_features, "ML")

    encoded = {}
    for col in feature_columns:
        if col in raw.columns:
            encoded[col] = raw[col].map(encode_genotype)
        else:
            encoded[col] = np.full(len(raw), np.nan)
    X = pd.DataFrame(encoded, columns=feature_columns)

    proba = np.asarray(model.predict_proba(X), dtype=float)
    predicted = np.asarray(classes)[np.argmax(proba, axis=1)]
    max_probability = np.nanmax(proba, axis=1)
    matched_count = len(present_features)
    missing_count = len(missing_features)

    sample_col = raw.columns[0]
    headers = (
        ["sample_id"]
        + classes
        + [
            "predicted_label",
            "max_probability",
            "matched_features_count",
            "missing_features_count",
        ]
    )
    rows = []
    for idx, (_, sample_row) in enumerate(raw.iterrows()):
        sample_id = str(sample_row[sample_col]).strip()
        rows.append(
            [sample_id]
            + [proba[idx, cls_idx] for cls_idx in range(len(classes))]
            + [
                predicted[idx],
                max_probability[idx],
                matched_count,
                missing_count,
            ]
        )

    summary = {
        "mode": "ML",
        "range": range_name,
        "algorithm": algorithm_alias,
        "algorithm_key": algorithm_key,
        "model_path": model_path,
        "feature_count": len(feature_columns),
        "input_marker_count": len(input_markers),
        "matched_feature_count": matched_count,
        "missing_feature_count": missing_count,
        "missing_feature_names": missing_features,
        "sample_count": len(raw),
        "classes": classes,
    }
    return headers, rows, summary


def format_output_value(value):
    """Format one cell for CSV output."""
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if value == 0:
            return "0"
        return f"{value:.8g}"
    return str(value)


def write_result(headers, rows, output_path):
    """Write a result table as CSV."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter=",", lineterminator="\n")
        writer.writerow(headers)
        for row in rows:
            writer.writerow([format_output_value(value) for value in row])


def format_terminal_value(value):
    """Format one cell for the terminal table."""
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.6g}"
    return str(value)


def render_terminal_table(headers, rows):
    """Render a simple aligned ASCII table for the command line."""
    formatted = [[format_terminal_value(value) for value in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in formatted:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def render_line(cells):
        return "| " + " | ".join(
            str(cell).ljust(widths[idx]) for idx, cell in enumerate(cells)
        ) + " |"

    lines = [
        separator,
        render_line(headers),
        separator,
    ]
    for row in formatted:
        lines.append(render_line(row))
    lines.append(separator)
    return "\n".join(lines)
