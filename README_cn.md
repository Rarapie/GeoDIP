# GeoDIP

GeoDIP 命令行预测工具，将原始 LR 方法与 5C/EAS 机器学习模型封装为统一的命令行
软件。用户提供一份基因型文件，即可完成群体来源预测，并在命令行中直接查看结果。

English version: [README.en.md](./README.en.md)

## 功能

- 支持两种预测模式：`LR`（似然比 / PAMP）和 `ML`（机器学习分类）。
- 支持两种预测范围：`5C`（非洲、美洲、东亚、欧洲、南亚）和
  `EAS`（汉、日本、东南亚）。
- 频率数据、全部模型与示例输入已打包到 `data/`，程序只从 `data/` 读取数据。
- LR 模式使用 `data/freq/freq_5c.csv` 与 `data/freq/freq_eas.csv`，
  按原 LR.py 的方法计算各群体 PAMP、LR 与预测标签。
- ML 模式自动加载 `data/models/{5C,EAS}/*.pkl`，
  输出各群体分类概率与预测标签。
- ML 模式支持 9 种算法，5C 与 EAS 的默认算法均为 `XGB`。
- 自动校验输入文件格式、样本 ID、基因型编码，并统计匹配位点数量。
- 结果表格固定输出为 CSV，并在终端中渲染表格。

## 环境要求

- Python 3.9+
- 依赖库：`pandas`、`numpy`、`scikit-learn`、`xgboost`、`joblib`

安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

在软件目录内运行：

```bash
# LR 模式，5C 范围
python geodip.py \
  predictor=LR range=5C \
  input=data/example/9948_geno.csv \
  output=result_5c_lr.csv

# LR 模式，EAS 范围
python geodip.py \
  predictor=LR range=EAS \
  input=data/example/9948_geno.csv \
  output=result_eas_lr.csv

# ML 模式，5C 范围，默认 XGB
python geodip.py \
  predictor=ML range=5C \
  input=data/example/9948_geno.csv \
  output=result_5c_ml.csv

# ML 模式，EAS 范围，指定 SVM
python geodip.py \
  predictor=ML range=EAS algorithm=SVM \
  input=data/example/9948_geno.csv \
  output=result_eas_ml.csv
```

短参数形式：

```bash
python geodip.py -p ML -r 5C -a XGB -i input.csv -o result.csv
```

输入文件也可以作为位置参数传入：

```bash
python geodip.py predictor=ML range=5C input.csv -o result.csv
```

## 参数说明

| 参数 | 短参数 | 必填 | 可选值 / 说明 |
| --- | --- | --- | --- |
| `predictor` | `p` | 是 | `LR` 或 `ML` |
| `range` | `r` | 是 | `5C` 或 `EAS` |
| `input` | `i` | 是 | 基因型文件路径，CSV 或 TSV |
| `output` | `o` | 是 | 结果输出文件路径 |
| `algorithm` | `a` | 否 | 仅 ML 模式；默认 `XGB` |
| `help` | `h` | 否 | 显示帮助 |

算法与模型文件对应关系：

| 参数值 | 模型 pkl |
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

## 输入文件格式

输入文件第一列必须是样本 ID 列，其余列为位点（marker）列。

```csv
Sample Name,rs2307840,Amelogenin,rs35785693,...
9948-PC,0|0,"X, Y",0|0,...
```

支持的基因型编码：

- 双等位编码：`0|0`、`0|1`、`1|0`、`1|1`
- 斜杠编码：`0/0`、`0/1`、`1/0`、`1/1`
- 缺失值：空值、`.`、`-`、`?`、`NA`、`N/A`、`NaN`

说明：

- 输入列可以多于模型特征列，多余的列会被忽略（例如 `Amelogenin`、
  `rs2032678`）。
- 模型缺少的位点会被自动记为缺失，由模型管道中的均值填充处理。
- 程序会报告输入位点数、频率/模型位点数、匹配位点数与缺失位点数。

## 输出格式

### LR 模式

5C 输出列：

```text
sample_id, PAMP_AFR, PAMP_AMR, PAMP_EAS, PAMP_EUR, PAMP_SAS,
LR, predicted_pop, valid_loci_count
```

EAS 输出列：

```text
sample_id, PAMP_HAN, PAMP_JPT, PAMP_SEAS,
LR, predicted_pop, valid_loci_count
```

### ML 模式

输出列：

```text
sample_id, <各群体概率列...>, predicted_label,
max_probability, matched_features_count, missing_features_count
```

结果固定输出为 CSV。运行结束后结果表格会同时打印在命令行窗口中。

## 目录结构

```text
GeoDIP/
├── geodip.py            # 命令行入口
├── core.py              # 输入校验、LR/ML 预测与表格输出
├── model_classes.py     # XGBoost 模型反序列化所需的包装类
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
