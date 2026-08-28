# GeoDIP

GeoDIP is a command line tool that packages the original likelihood-ratio
method and the 5C/EAS machine learning models. Given a
genotype table, it predicts population origin and prints a result table
directly in the terminal.

Chinese version: [README_cn.md](./README_cn.md)

## Features

- Two prediction modes: `LR` (likelihood ratio / PAMP) and `ML` (machine
  learning classification).
- Two prediction ranges: `5C` (Africa, Americas, East Asia, Europe, South
  Asia) and `EAS` (Han, Japanese, Southeast Asian).
- Frequency tables, all trained models, and the example input are bundled
  under `data/`. The program reads data from `data/` only.
- LR mode uses `data/freq/freq_5c.csv` and
  `data/freq/freq_eas.csv` and follows the original LR.py PAMP/LR
  calculation.
- ML mode loads `data/models/{5C,EAS}/*.pkl` and outputs class
  probabilities plus the predicted label.
- Nine ML algorithms are supported; the default algorithm is `XGB` for both
  `5C` and `EAS`.
- Input format, sample IDs, and genotype calls are validated, and the number
  of matching markers is reported.
- Result tables are always written as CSV and rendered in the terminal.

## Requirements

- Python 3.9+
- `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Run from the software directory:

```bash
# LR mode, 5C range
python geodip.py \
  predictor=LR range=5C \
  input=data/example/9948_geno.csv \
  output=result_5c_lr.csv

# LR mode, EAS range
python geodip.py \
  predictor=LR range=EAS \
  input=data/example/9948_geno.csv \
  output=result_eas_lr.csv

# ML mode, 5C range, default XGB
python geodip.py \
  predictor=ML range=5C \
  input=data/example/9948_geno.csv \
  output=result_5c_ml.csv

# ML mode, EAS range, SVM
python geodip.py \
  predictor=ML range=EAS algorithm=SVM \
  input=data/example/9948_geno.csv \
  output=result_eas_ml.csv
```

Short option form:

```bash
python geodip.py -p ML -r 5C -a XGB -i input.csv -o result.csv
```

The input file can also be passed as a positional argument:

```bash
python geodip.py predictor=ML range=5C input.csv -o result.csv
```

## Arguments

| Argument | Short | Required | Values / Description |
| --- | --- | --- | --- |
| `predictor` | `p` | Yes | `LR` or `ML` |
| `range` | `r` | Yes | `5C` or `EAS` |
| `input` | `i` | Yes | Genotype file path, CSV or TSV |
| `output` | `o` | Yes | Output result file path |
| `algorithm` | `a` | No | ML only; default `XGB` |
| `help` | `h` | No | Show help |

Algorithm names and model files:

| Argument | Model file |
| --- | --- |
| `LR` | `logistic_regression.pkl` |
| `GNB` | `naive_bayes.pkl` |
| `KNN` | `knn.pkl` |
| `SVM` | `svm.pkl` |
| `RF` | `random_forest.pkl` |
| `HGB` | `gradient_boosting.pkl` |
| `XGB` | `xgboost.pkl` |
| `AGB` | `adaboost.pkl` |
| `MLP` | `mlp.pkl` |

## Input Format

The first column must be the sample ID column; all other columns are markers.

```csv
Sample Name,rs2307840,Amelogenin,rs35785693,...
9948-PC,0|0,"X, Y",0|0,...
```

Supported genotype encodings:

- Diploid pairs: `0|0`, `0|1`, `1|0`, `1|1`
- Slash notation: `0/0`, `0/1`, `1/0`, `1/1`
- Missing values: empty, `.`, `-`, `?`, `NA`, `N/A`, `NaN`

Guidelines:

- Extra input columns are ignored when they are not model features (for
  example `Amelogenin` or `rs2032678`).
- Missing model features are filled by the model pipeline's mean imputer.
- The tool reports input markers, model/frequency markers, matched markers,
  and missing markers.

## Output Format

### LR Mode

5C columns:

```text
sample_id, PAMP_AFR, PAMP_AMR, PAMP_EAS, PAMP_EUR, PAMP_SAS,
LR, predicted_pop, valid_loci_count
```

EAS columns:

```text
sample_id, PAMP_HAN, PAMP_JPT, PAMP_SEAS,
LR, predicted_pop, valid_loci_count
```

### ML Mode

Columns:

```text
sample_id, <class probability columns...>, predicted_label,
max_probability, matched_features_count, missing_features_count
```

Results are always written as CSV. The result table is also printed in the
terminal after a run.

## Directory Structure

```text
GeoDIP/
├── geodip.py            # Command line entry point
├── core.py              # Validation, prediction, and table output
├── model_classes.py     # XGBoost wrapper needed to unpickle model files
├── data/
│   ├── freq/
│   │   ├── freq_5c.csv
│   │   └── freq_eas.csv
│   ├── models/
│   │   ├── 5C/*.pkl
│   │   └── EAS/*.pkl
│   └── example/
│       └── 9948_geno.csv
├── requirements.txt
├── README.en.md
└── README.md
```
