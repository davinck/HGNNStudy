# HGNNStudy：DHG-Bench 节点分类复现

本仓库复现 **DHG-Bench (ICLR 2026)** 的节点分类（`node_cls`）任务，覆盖论文 Table 1 与 Table A5 中的 **20 种方法 × 13 个数据集 = 260 个配置**。仓库内容分四部分：上游官方代码（`DHG-Bench/`，未改动）、数据集（`DHG-Bench/data/`，本项目中未上传）、复现报告（`Result/`，LaTeX 源与编译产物）、论文阅读笔记（`note/`）。

本仓库的代码与数据来自 DHG-Bench，引用如下：
```bibtex
@inproceedings{li2026dhg,
  title     = {DHG-Bench: A Comprehensive Benchmark for Deep Hypergraph Learning},
  author    = {Li, Fan and Wang, Xiaoyang and Zhang, Wenjie and Zhang, Ying and Lin, Xuemin},
  booktitle = {International Conference on Learning Representations},
  volume    = {2026},
  pages     = {47693--47727},
  year      = {2026}
}
```

| 项目   | 状态                                                                                                                              |
| ---- | ------------------------------------------------------------------------------------------------------------------------------- |
| 上游代码 | `DHG-Bench/dhgbench/`，来源 <https://github.com/Coco-Hut/DHG-Bench>，复现过程中未改动其代码                                                    |
| 数据集  | `DHG-Bench/data/` 本项目中未上传（5 个子目录，约 535 MB）来源<https://huggingface.co/datasets/Philley/DHG-Bench-dataset>，首次运行自动生成 `pyg_data/` 缓存 |
| 复现规模 | 260 格中 **233 格跑通**、27 格 OOM；可对比数值 228 格                                                                                         |
| 复现精度 | 平均 \|Δ\| = **0.53** 个百分点，中位数 **0.15**，203/228（89.0%）落在 1 个百分点以内                                                                 |
| 报告   | `Result/main.pdf`（XeLaTeX + biber 编译）                                                                                           |

四项任务的交付位置：

| 任务     | 内容                                | 位置                                                           |
| ------ | --------------------------------- | ------------------------------------------------------------ |
| Task 1 | 20 种方法 × 13 个数据集的复现结果             | `Result/task_1_Reproduction.tex`（13 张表）                      |
| Task 2 | 数据集统计信息                           | `Result/task_2_Data.tex`；本文 2.2 节                            |
| Task 3 | 20 种 baseline 方法介绍与引用             | `Result/task_3_Baseline.tex`；引用链接见本文第 7 章；读论文的过程性笔记在 `note/` |
| Task 4 | 个人 GitHub 仓库（代码 + 数据 + 报告 + 复现说明） | 本文件                                                          |

**声明**
- `DHG-Bench/` 下的代码、数据与 `README.md` 版权归原作者所有，本仓库仅用于课程学习与复现，复现过程中未改动上游代码。
- 上游仓库根目录未附带 LICENSE 文件，因此本仓库不对 `DHG-Bench/` 的内容授予任何许可；数据集本身在原发布页标注为 MIT 许可。
- `Result/` 下的报告、各类表格与统计由本人完成。
---

## 目录

- [1. 环境安装与数据集下载](#1-环境安装与数据集下载)
- [2. 项目结构](#2-项目结构)
- [3. 运行方法](#3-运行方法)
- [4. 附录](#4-附录)
- [5. 算法引用](#5-算法引用)

> **路径约定**：以下所有命令默认在仓库根目录 `HGNNStudy/`（即同时含 `DHG-Bench/` 与 `Result/` 的那一级）执行。训练必须在 `DHG-Bench/dhgbench/` 下启动，因为代码内部统一用 `../data/...` 相对路径读取数据。

---

## 1. 环境安装与数据集下载

本机实测环境：Windows 11 + NVIDIA GeForce RTX 3060 Laptop（6 GB）+ 驱动 617.42，Python 3.10.22 / PyTorch 2.2.2+cu121 / PyTorch Geometric 2.6.1。安装顺序不能换，下面按四步写完，最后给出装好后的完整包清单。

### 1.1 配置环境

**第一步，创建环境**
```bash
conda create -n dhg python=3.10 -y
conda activate dhg
python -V                      # 必须是 3.10.x
```

**第二步，安装依赖** 
```bash

pip install torch==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121

pip install torch_geometric==2.6.1
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.htmlls

pip install "numpy==1.24.3" pyyaml dask tqdm pandas scipy scikit-learn
```

**第三步，验证。** 
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import torch_scatter, torch_sparse; print('pyg-ext ok')"
python -c "import torch_geometric, numpy; print(torch_geometric.__version__, numpy.__version__)"
cd DHG-Bench/dhgbench && python -c "from lib_utils.exp_agent import ExpAgent; print('project ok')"
```

期望输出：
```
2.2.2+cu121 True NVIDIA GeForce RTX 3060 Laptop GPU
pyg-ext ok
2.6.1 1.24.3
project ok
```

**关键依赖**（真正需要显式安装的 11 个包）：

| 包                 | 版本               | 作用                                         |
| ----------------- | ---------------- | ------------------------------------------ |
| `torch`           | 2.2.2+cu121      | 深度学习框架（CUDA 12.1 构建）                       |
| `torch-geometric` | 2.6.1            | 图 / 超图消息传递框架                               |
| `torch_scatter`   | 2.1.2+pt22cu121  | 分段聚合算子，`lib_dataset/preprocessing.py` 直接依赖 |
| `torch_sparse`    | 0.6.18+pt22cu121 | 稀疏矩阵算子                                     |
| `numpy`           | 1.24.3           | 数值计算，必须锁 1.x                               |
| `scipy`           | 1.15.3           | 稀疏矩阵与科学计算                                  |
| `scikit-learn`    | 1.7.2            | 数据集划分与评估指标                                 |
| `pandas`          | 2.3.3            | 表格读取与统计输出                                  |
| `PyYAML`          | 6.0.3            | 读取 `lib_yamls/*.yaml` 超参配置                 |
| `dask`            | 2026.8.0         | 大图数据的分块处理                                  |
| `tqdm`            | 4.70.1           | 训练进度条                                      |

所有依赖及对应版本均在 **requirements.txt** 中记录，在项目终端中运行 `pip install -r requirements.txt`，即可按照文件中指定的依赖及版本自动安装所需环境

### 1.2 数据集下载

**第一步，安装下载工具**
```
pip install -U huggingface_hub
```

**第二步，下载数据集**
```
hf download Philley/DHG-Bench-dataset data.zip --repo-type dataset --local-dir ./datasets/DHG-Bench-dataset
```

**第三步，解压数据集**
```
import zipfile

zip_path = "datasets/DHG-Bench-dataset/data.zip"
extract_path = "datasets/DHG-Bench-dataset/"

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print("数据集解压完成！")
```

---

## 2. 项目结构

```
HGNNStudy/
├── DHG-Bench/                    # 上游官方代码与数据（未改动）
│   ├── dhgbench/                 #   训练入口 main.py、20 种方法实现、YAML 配置
│   │   ├── lib_models/           #   各方法模型实现
│   │   ├── lib_dataset/          #   数据加载与预处理
│   │   ├── lib_yamls/node_yamls/ #   ★ 逐数据集的超参配置文件
│   │   └── lib_utils/            #   训练、评估、日志
│   ├── data/                     # 数据集（约 535 MB，见 2.2）
│   ├── resources/                # 上游 README 的图与 logo
│   ├── tests/                    # 上游单元测试
│   └── README.md                 # 上游原始说明（保留）
├── Result/                       # 复现报告
│   ├── main.tex                  #   主文件，\input 四个任务
│   ├── task_1_Reproduction.tex   #   Task 1：13 张结果表（260 格）
│   ├── task_2_Data.tex           #   Task 2：13 个数据集统计
│   ├── task_3_Baseline.tex       #   Task 3：20 种 baseline 介绍
│   ├── task_4_info.tex           #   Task 4：仓库信息
│   ├── refs.bib                  #   参考文献
│   └── main.pdf                  #   编译产物
├── note/                         # 论文阅读笔记（当前为空，后续补充）
└── README.md
```

### 2.1 代码：来自上游，未做修改

`DHG-Bench/` 下的 `dhgbench/`、`data/`、`resources/`、`tests/` 全部来自官方仓库 <https://github.com/Coco-Hut/DHG-Bench>。复现过程中**没有修改任何模型实现或配置文件**——这是复现而非调参的前提。

### 2.2 数据：13 个节点分类数据集

数据已解压到 `DHG-Bench/data/`（与 `dhgbench/` 同级，约 535 MB）。首次运行某个数据集时，会在 `data/pyg_data/hypergraph_dataset_updated/` 下自动生成 `.pt` 缓存，之后复用。

统计量由脚本从原始 pickle 直接读出，与论文 Table A1 逐项一致：

| 数据集        | `--dname`           |     节点数 |     超边数 |  特征维度 | 类别数 | 平均\|e\| |
| ---------- | ------------------- | ------: | ------: | ----: | --: | ------: |
| Cora       | `cora`              |   2,708 |   1,579 | 1,433 |   7 |    3.03 |
| Pubmed     | `pubmed`            |  19,717 |   7,963 |   500 |   3 |    4.35 |
| Cora-CA    | `coauthor_cora`     |   2,708 |   1,072 | 1,433 |   7 |    4.28 |
| DBLP-CA    | `coauthor_dblp`     |  41,302 |  22,363 | 1,425 |   6 |    4.45 |
| Walmart    | `walmart-trips-100` |  88,860 |  69,906 |    11 |  11 |    6.59 |
| Trivago    | `trivago`           | 172,738 | 233,202 |   300 | 160 |    3.12 |
| Actor      | `actor`             |  16,255 |  10,164 |    50 |   3 |    5.25 |
| Gamers     | `twitch`            |  16,812 |   2,627 |     7 |   2 |    6.23 |
| Pokec      | `pokec`             |  14,998 |   2,406 |    65 |   2 |    2.29 |
| Yelp       | `yelp`              |  50,758 | 679,302 | 1,862 |   9 |    6.66 |
| NTU2012    | `NTU2012`           |   2,012 |   2,012 |   100 |  67 |    5.00 |
| ModelNet40 | `ModelNet40`        |  12,311 |  12,311 |   100 |  40 |    5.00 |
| Ratings    | `amazon`            |  22,299 |   2,090 |   111 |   5 |    3.10 |

两点说明：

- **Walmart** 原始特征为 11 维（11 个类别）的 one-hot，加载后扩展到 100 维；PhenomNN 只支持 one-hot 输入，故该列按原始维度计。
- `data/` 下另有 `fair_data/`、`hgcls_data/` 等目录，属于数据包附带的其他任务数据集，本复现**未使用**。

### 2.3 报告：`Result/`

四个任务各一个 `.tex`，由 `main.tex` 统一 `\input`，`main.pdf` 为编译结果。260 格的完整数值在 `task_1_Reproduction.tex` 中，按数据集分为 13 张表。

### 2.4 笔记：`note/`

`note/` 存放读论文时的笔记：DHG-Bench 原文的要点、20 种 baseline 各自原论文的方法与实验细节、以及复现过程中想不通的地方。当前目录为空，后续逐篇补充。

---

## 3. 运行方法

### 3.1 快速开始

```bash
# ① 激活环境
conda activate dhg

# ② 进入训练目录（必须，代码用 ../data 相对路径）
cd DHG-Bench/dhgbench

# ③ 单格：HGNN @ Cora，5 个种子，GPU 上约 10 秒
python main.py --dname=cora --task_type=node_cls --method=HGNN --num_seeds=5

# ④ 换数据集 / 换方法只需改两个参数，例如 TF-HNN @ Pubmed
python main.py --dname=pubmed --task_type=node_cls --method=TFHNN --num_seeds=5
```

运行结束时打印 `[Final]` 段，形如：

```
test_acc: 78.1400+-1.19
```

即 **78.14±1.19%**——均值与标准差**都是百分数**（`lib_utils/metrics.py` 中的 `masked_accuracy` 已经乘过 100），可直接写进表格。

### 3.2 复现命令

所有实验结果都来自同一条命令模板，只换 `--dname` 与 `--method` 两个参数：

```bash
python main.py --dname=cora --task_type=node_cls --method=HGNN --num_seeds=5
```

各参数的含义：

- `--dname=cora`
  - 作用：指定数据集。`cora` 是经典引文网络数据集；13 个数据集与代码名的对应见 2.2 节，全部可选值在 `lib_dataset/__init__.py` 的 `_single_datasets_` 中
```
数据集
	['cora', 'pubmed', 'coauthor_cora', 'coauthor_dblp','walmart-trips-100', 'trivago','actor', 'twitch', 'pokec', 'yelp', 'NTU2012', 'ModelNet40']
```
- `--task_type=node_cls`
  - 作用：指定任务类型。`node_cls` 为节点分类，是本仓库全部实验所属的任务；框架另支持 `edge_pred`（边预测）与 `hg_cls`（超图分类）。该参数**默认是 `edge_pred`**，漏写会静默跑到边预测分支上。
- `--method=HGNN`
  - 作用：指定模型。`HGNN` 即超图神经网络；20 个可选值在 `lib_utils/exp_agent.py` 的 `parse_model` 中，其中 5 个写法与论文不同（`TFHNN`、`UniGIN`、`AllSetformer`、`EDHNN`、`TMPHN`）。
```
模型
	['MLP', 'CEGCN','CEGAT', 'HGNN', 'HyperGCN', 'HCHA', 'LEGCN', 'HyperND', PhenomNN', 'SheafHyperGNN', 'HJRL', 'DPHGNN', 'TFHNN', 'HNHN', 'UniGCNII', 'AllSetformer', 'EDHNN', 'HyperGT', 'EHNN','TMPHN']
```
- `--num_seeds=5`
  - 作用：指定重复次数（随机种子数量）。设为 5 表示依次跑种子 0–4 共 5 次，最后输出 5 次的均值与标准差。该参数**默认只有 2**，必须显式指定，否则标准差只由 2 次结果算出、波动很大。
-  `--is_default（注意）`
	- 每个方法的超参写在 `DHG-Bench/dhgbench/lib_yamls/node_yamls/config_<方法>.yaml`，按数据集分段。程序启动时按 `--dname` 取对应段落覆盖 argparse 默认值
	- 不要加 `--is_default=True`。该参数会改为读取 YAML 的 `default` 段，把作者按数据集调好的超参全部丢掉，结果会明显偏低。只有在 YAML 确实没有该数据集段落时才需要它。(模型没有对应数据集的超参数见4.1)
	- **`--is_default` 没有布尔解析器**。写 `--is_default=False` 得到的是非空字符串，在 `if` 里为真，等于把开关打开了。只能不写，或写 `--is_default=True`。


其余超参写死在代码里，命令行改不动：

- **数据划分**：随机 50% / 25% / 25%（训练 / 验证 / 测试），全图节点都参与——`set_task_args` + `generate_random_split`。
- **指标**：测试准确率的均值 ± 标准差（%），`masked_accuracy` 中已乘 100。
- **损失与优化器**：负对数似然（NLL）+ Adam——`train_agent.py`。
- **训练方式**：全批量（full-batch），不划分 mini-batch——`train_agent.py`。
- **早停**：关闭，直接取最后一轮的测试结果——`set_task_args`。
- **自环**：开启，每个节点补一条单元素超边——`node_cls` 分支里硬编码 `add_self_loop=True`。

### 3.3 编译报告

```bash
cd Result
xelatex main.tex && biber main && xelatex main.tex && xelatex main.tex
```

需要 TeX Live / MiKTeX（含 ctex、biblatex-gb7714-2015）。注意 `main.tex` 指定的程序是 `xelatex`，用 `pdflatex` 会在中文字体与 `ctexart` 处报错。

---


## 4. 附录

### 4.1 YAML 未提供配置的 30 个组合

下列组合在 `lib_yamls/node_yamls/` 中没有对应数据集段落，程序会退回 argparse 默认值（`epochs=5` 等），结果为「非调参」状态，参考价值有限。

| 方法            | 缺失的数据集                                                                                    |  格数 |
| ------------- | ----------------------------------------------------------------------------------------- | --: |
| CEGAT         | Trivago、Yelp                                                                              |   2 |
| LEGCN         | Yelp                                                                                      |   1 |
| PhenomNN      | Trivago                                                                                   |   1 |
| SheafHyperGNN | Walmart、Trivago、Yelp                                                                      |   3 |
| HJRL          | DBLP-CA、Walmart、Trivago、Yelp                                                              |   4 |
| DPHGNN        | Walmart、Trivago、Yelp                                                                      |   3 |
| UniGNN        | Pubmed、Cora-CA、DBLP-CA、Walmart、Trivago、Actor、Gamers、Pokec、Yelp、NTU2012、ModelNet40、Ratings |  12 |
| HyperGT       | Yelp                                                                                      |   1 |
| T-HyperGNN    | Walmart、Trivago、Yelp                                                                      |   3 |

### 4.2 常见问题

| 现象                                  | 原因                      | 处理                                                             |
| ----------------------------------- | ----------------------- | -------------------------------------------------------------- |
| `Unimplemented model <name>`        | `--method` 用了论文写法       | 换成第 5 章开头列出的代码名（TFHNN / UniGIN / AllSetformer / EDHNN / TMPHN） |
| `FileNotFoundError: ../data/...`    | 不在 `dhgbench/` 下启动      | `cd DHG-Bench/dhgbench` 后再跑                                    |
| `torch.cuda.is_available()` 为 False | 装了 CPU 版 torch          | 重装 `+cu121` 轮子，见第 1 章第二步                                       |
| `Failed to initialize NumPy`        | numpy 2.x               | `pip install numpy==1.24.3`                                    |
| 结果比论文低 2–5 个点                       | 误加了 `--is_default=True` | 去掉该参数                                                          |
| `--epochs` 改了没反应                    | YAML 覆盖命令行              | 改 YAML，或接受默认轮数                                                 |
| `CUDA out of memory`                | 6 GB 显存不足               | 换 `--device=cpu`，或记为 OOM                                       |

## 5. 算法引用

表中题名为原论文标题的中文译名，点击即可跳转原文；链接均经 Crossref / arXiv / PMLR 回查核对。

| ID | 方法 | 论文 | 场地 |
|---:|---|---|---|
| 1 | HGNN | [超图神经网络](https://doi.org/10.1609/aaai.v33i01.33013558) | AAAI 2019 |
| 2 | HyperGCN | [HyperGCN：在超图上训练图卷积网络的新方法](https://arxiv.org/abs/1809.02589) | NeurIPS 2019 |
| 3 | HNHN | [HNHN：带超边神经元的超图网络](https://arxiv.org/abs/2006.12278) | ICML Workshop 2020 |
| 4 | HCHA | [超图卷积与超图注意力](https://doi.org/10.1016/j.patcog.2020.107637) | Pattern Recognition 2021 |
| 5 | UniGNN | [UniGNN：图与超图神经网络的统一框架](https://doi.org/10.24963/ijcai.2021/353) | IJCAI 2021 |
| 6 | AllSetTransformer | [AllSet：超图神经网络的多重集函数框架](https://arxiv.org/abs/2106.13264) | ICLR 2022 |
| 7 | HyperND | [超图上的非线性特征扩散](https://proceedings.mlr.press/v162/prokopchik22a.html) | ICML 2022 |
| 8 | EHNN | [等变超图神经网络](https://arxiv.org/abs/2208.10428) | ECCV 2022 |
| 9 | LEGCN | [基于超图线展开的半监督超图节点分类](https://doi.org/10.1145/3511808.3557447) | CIKM 2022 |
| 10 | ED-HNN | [等变超图扩散神经算子](https://arxiv.org/abs/2207.06680) | ICLR 2023 |
| 11 | PhenomNN | [从超图能量函数到超图神经网络](https://arxiv.org/abs/2306.09623) | ICML 2023 |
| 12 | SheafHyperGNN | [层化（Sheaf）超图网络](https://doi.org/10.52202/075280-0529) | NeurIPS 2023 |
| 13 | HJRL | [基于交叉扩展的超顶点与超边联合表示学习](https://doi.org/10.1609/aaai.v38i8.28775) | AAAI 2024 |
| 14 | DPHGNN | [DPHGNN：双视角超图神经网络](https://doi.org/10.1145/3637528.3672047) | KDD 2024 |
| 15 | T-HyperGNN | [T-HyperGNNs：基于张量表示的超图神经网络](https://doi.org/10.1109/tnnls.2024.3371382) | IEEE TNNLS 2024 |
| 16 | HyperGT | [面向半监督分类的超图 Transformer](https://doi.org/10.1109/icassp48485.2024.10446248) | ICASSP 2024 |
| 17 | TF-HNN | [用于超图学习的免训练消息传递](https://arxiv.org/abs/2402.05569) | ICLR 2025 |
| 18 | MLP | 多层感知机基线（不使用超图结构） | — |
| 19 | CEGCN | [基于超图的学习：聚类、分类与嵌入（团展开 + GCN）](https://doi.org/10.7551/mitpress/7503.003.0205) | NeurIPS 2006 |
| 20 | CEGAT | [基于超图的学习：聚类、分类与嵌入（团展开 + GAT）](https://doi.org/10.7551/mitpress/7503.003.0205) | NeurIPS 2006 |

说明：

- 前 17 种为论文 Table 1 / Table A5 中的超图神经网络方法；
- **MLP** 不使用结构信息，无对应原创论文，不附链接；
- **CEGCN / CEGAT** 并非独立论文，而是把超图按团展开（clique expansion）退化为普通图后分别套用 GCN 与 GAT，故共同引用 Zhou et al. 2006；
- 部分会议论文（NeurIPS / ICML / ICLR / ECCV）无 DOI，链接指向 arXiv 预印本或 PMLR 正式论文集；
- ACM、MIT Press 等出版社对脚本访问返回 403，用浏览器打开正常。

