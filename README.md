# Retro-R1 逆合成路径规划

本仓库基于 [Retro*](https://github.com/binghong-ml/retro_star) 和 [Agent-R1](https://github.com/0russwest0/Agent-R1) 的开源代码构建。

## 环境配置

### 创建 conda 环境
```bash
git clone -b validation_API --single-branch https://github.com/elated-sawyer/Agent-r1-temp.git 
cd Retro-R1
conda env create --file environment.yml  # 不一定需要完整安装此环境（无关 package 较多），可以直接使用已有的能调用 API 的环境，运行时缺什么包再补装即可（大约 3～4 个化学相关的包）
conda activate Retro_R1
pip install -e packages/mlp_retrosyn
pip install -e packages/rdchiral
unzip verl.zip
cd verl
pip install --no-deps -e .
```

### 下载所需文件
为了复现论文结果，需要额外下载训练数据集、评测数据集（USPTO、ChEMBL-1000）、起始分子库以及模板规则：
- **Retro\* 原始文件**（含单步模型权重 V1）：从[此链接](https://www.dropbox.com/s/ar9cupb18hv96gj/retro_data.zip?dl=0)下载，将 `dataset/` 和 `one_step_model/` 文件夹放在项目根目录下。
- **单步模型权重 V2 / V3**：从[此链接](https://drive.google.com/drive/u/0/folders/13DdftEV0x55OZ8ZxHNAkmcvi_4x90hPI)下载，将 `retro_star_value_ours.ckpt`（V2）和 `retro_star_zero_ours.ckpt`（V3）放入 `one_step_model/` 目录。
- **单步模型权重 V4**：找我要。权重命名为 `retro_star_V4.ckpt` 并放入 `one_step_model/` 目录。
- **ChEMBL-1000 测试集**：从[此链接](https://drive.google.com/drive/folders/198WuPlSyMeMvvd4i2SM833jPAcGllzDu?usp=sharing)下载，放入 `dataset/` 目录。

## 数据预处理
```bash
python ./examples/data_preprocess/reaction.py
```

## 评测
```bash
sbatch --export=ALL,DATASET=retro,MAX_TURNS=100,FORCE_NOLOOP=True,API_MODEL_NAME=Qwen/Qwen3.5-35B-A3B test.sbatch
```

可配置参数：
| 参数 | 可选值 | 说明 |
|------|--------|------|
| `DATASET` | `retro` / `chembl` | 评测数据集 |
| `MAX_TURNS` | 整数，默认 `100` | 最大对话轮次（不用修改） |
| `FORCE_NOLOOP` | `True` / `False` | 是否启用 Pass@1 无悔搜索模式 |
| `API_MODEL_NAME` | 模型名称 | 要评测的 API 模型 |

## 结果记录

https://aicarrier.feishu.cn/wiki/PC2Zw67kki0g3GkSDJrcfbJMngg?larkTabName=space#share-V3kBdK0Swo0pBnxFnZrcWtSQnch

---

# Retro-R1 for Retrosynthetic Planning

This repository is based on the open-source codebase of [Retro*](https://github.com/binghong-ml/retro_star) and [Agent-R1](https://github.com/0russwest0/Agent-R1).

## Setup

### Create conda environment
```bash
git clone -b validation_API --single-branch https://github.com/elated-sawyer/Agent-r1-temp.git 
cd Retro-R1
conda env create --file environment.yml  # Not necessarily needed (many unused packages). You can use an existing env that supports API calls, and install 3-4 chemistry packages as needed when errors occur.
conda activate Retro_R1
pip install -e packages/mlp_retrosyn
pip install -e packages/rdchiral
unzip verl.zip
cd verl
pip install --no-deps -e .
```

### Download the necessary files
To reproduce the results in the paper, we also need the additional files containing the training dataset, evaluation datasets (USPTO, ChEMBL-1000), starting molecules, and the template rules.
- **Retro\* files** (including single-step model weights V1): download from [this link](https://www.dropbox.com/s/ar9cupb18hv96gj/retro_data.zip?dl=0), and put `dataset/` and `one_step_model/` under the project root.
- **Single-step model weights V2 / V3**: download from [this link](https://drive.google.com/drive/u/0/folders/13DdftEV0x55OZ8ZxHNAkmcvi_4x90hPI), and put `retro_star_value_ours.ckpt` (V2) and `retro_star_zero_ours.ckpt` (V3) under `one_step_model/`.
- **Single-step model weights V4**: not publicly released. Can be trained following [PDVN](https://github.com/DiXue98/PDVN) instructions, or provided upon request. Name it `retro_star_V4.ckpt` and place under `one_step_model/`.
- **ChEMBL-1000 testset**: download from [this link](https://drive.google.com/drive/folders/198WuPlSyMeMvvd4i2SM833jPAcGllzDu?usp=sharing) and put under `dataset/`.

## Preprocess the dataset
```bash
python ./examples/data_preprocess/reaction.py
```

## Evaluation
```bash
sbatch --export=ALL,DATASET=retro,MAX_TURNS=100,FORCE_NOLOOP=True,API_MODEL_NAME=Qwen/Qwen3.5-35B-A3B test.sbatch
```

Configurable parameters:
| Parameter | Values | Description |
|-----------|--------|-------------|
| `DATASET` | `retro` / `chembl` | Evaluation dataset |
| `MAX_TURNS` | integer, default `100` | Maximum conversation turns |
| `FORCE_NOLOOP` | `True` / `False` | Enable Pass@1 no-regret search mode |
| `API_MODEL_NAME` | model name | API model to evaluate |

## Results

https://aicarrier.feishu.cn/wiki/PC2Zw67kki0g3GkSDJrcfbJMngg?larkTabName=space#share-V3kBdK0Swo0pBnxFnZrcWtSQnch