# SGCN on OGBN-Proteins

本仓库基于 PyTorch Geometric（PyG）实现了 `ogbn-proteins` 的多标签节点分类实验，重点支持 **SGCN 子图训练**，并提供 OOD 扰动、结果记录与误差分析工具。

## 项目结构

```text
SGCN/
├── ogbn_proteins_pyg.ipynb      # 主入口：训练与实验流程（推荐）
├── install_a100_env.sh          # A100 环境一键安装脚本
├── A100_ENVIRONMENT.md          # A100 环境说明
├── dataset/                     # OGB 数据与映射文件目录
└── src/
    ├── models.py                # GNN 模型（gcn/sage/graphsaint/sgcn）
    ├── train.py                 # 训练与评估主逻辑（含 SGCN 聚合）
    ├── utils.py                 # 数据加载、预处理、OOD 扰动
    ├── logging_utils.py         # 实验日志与结果保存
    ├── evaluation.py            # True Path Rule 违规率计算
    ├── error_analysis.py        # FP/FN 离线错误分析
    ├── visualization.py         # 曲线与图表绘制
    └── build_go_parents.py      # GO 父子关系构建脚本
```

## 环境准备

### 方式 1：A100（推荐）

在仓库根目录执行：

```bash
bash install_a100_env.sh
conda activate sgcn-a100
```

更多参数见：`A100_ENVIRONMENT.md`

### 方式 2：手动安装（最小依赖）

```bash
pip install torch torch_geometric ogb pandas matplotlib pyyaml
```

> 说明：不同 CUDA / PyTorch 版本下，`torch_scatter`、`torch_sparse` 等 PyG 扩展需按官方说明选择对应 wheel。

## 快速开始（Notebook）

1. 打开 `ogbn_proteins_pyg.ipynb`
2. 在配置单元中设置关键超参数，例如：
   - `MPNN='sgcn'`（也支持 `gcn` / `graphsaint` / `sage`）
   - `N_SUBGRAPHS`, `LOCAL_EPOCHS`, `SUBSAMPLING_METHOD`
   - `OOD_ENABLE`, `Pood`, `Pcr`
3. 运行全部单元开始训练与评估。

默认会在 `results/exp_<timestamp>/` 下生成：

- `config.yaml`
- `run_summary.csv`
- `aggregate_summary.csv`
- `epoch_metrics.csv`
- `figures/*.png`
- `preds/*`（当 `SAVE_PRED=True` 时）

## 数据路径说明

当前 `src/utils.py` 中的 `load_data` 默认使用：

```python
PygNodePropPredDataset(name=dataset_name, root='/mnt/SGCN/dataset')
```

这是代码中的默认绝对路径，不要求必须使用该目录。  
如果你的数据目录不同（例如仓库内 `./dataset`），请修改该路径，或将数据准备到对应位置。

## 评估与分析脚本

### 1) True Path Rule (TPR) Violation

```bash
python src/evaluation.py --split test
```

常用参数：

- `--probs`：指定预测文件（`.pt/.pth/.npy/.csv`）
- `--logits`：输入为 logits 时自动做 sigmoid
- `--go-parents`：指定 GO 父子映射（默认 `dataset/ogbn_proteins/mapping/go_parents.csv`）
- `--json`：输出 JSON 统计结果

### 2) FP/FN 错误分析

```bash
python src/error_analysis.py --split test --threshold 0.5 --top-k 20
```

可选参数：

- `--detail`：导出逐样本 FP/FN 明细（TSV）
- `--max-items`：明细导出条数上限
- `--label-mapping`：标签索引到 GO term 映射文件

## 代码检查

当前仓库可使用以下基础语法检查：

```bash
python -m compileall src
```

## 许可证

仓库暂未声明许可证；如需开源分发，建议补充 `LICENSE` 文件。
