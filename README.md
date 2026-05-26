# Qwen Symbolic Regression — 对比算法独立运行包

本仓库包含 9 种符号回归（Symbolic Regression）对比算法的独立运行封装，与主实验仓库 `qwen_experiments/` 同级。

## 项目结构

```
qwen_experiments_deps/
├── wrappers/                    # 各基线的 Python wrapper
│   ├── gplearn/                 # gplearn 封装配置
│   ├── icsr/                    # In-Context Symbolic Regression
│   ├── llm_sr/                  # LLM-SR
│   ├── sr_llm/                  # SR-LLM
│   ├── lasr/                    # Library-Augmented Symbolic Regression
│   ├── pysr/                    # PySR
│   ├── gp_gomea/                # GP-GOMEA
│   └── dsr/                     # DSR / uDSR (Deep Symbolic Optimization)
├── gplearn/                     # gplearn 独立运行脚本（无 __init__.py）
├── utils/                       # 工具模块（自包含，独立于主实验）
├── external/                    # 上游代码库（需手动拉取，见 .gitignore）
├── benchmark_datasets/          # 基准数据集（fitness_cases, hold_out）
├── data/                        # GP 初始化池
├── run_all_baselines.py         # 统一启动脚本（可选）
├── requirements-*.txt           # 各基线的依赖文件
├── SETUP.md                     # 详细安装与运行指南
└── .env                         # API Key、模型配置（勿提交）
```

### 对比算法一览

| 算法 | 类型 | 资源消耗 | 是否需要 API Key | 是否需要外部运行时 |
|------|------|---------|-----------------|-------------------|
| **gplearn** | 传统 GP | CPU 低 | 否 | 无 |
| **ICSR** | LLM 驱动 | GPU/API | 是 (qwen3.5-plus) | 无 |
| **LLM-SR** | LLM 驱动 | API | 是 (qwen3.5-plus) | 无 |
| **SR-LLM** | LLM + RAG | GPU+API | 是 (qwen3.5-plus) | PyTorch |
| **LaSR** | LLM + Julia | API+Julia | 是 (qwen3.5-plus) | Julia |
| **PySR** | 遗传编程 | Julia | 否 | Julia |
| **GP-GOMEA** | 传统 GP | CPU | 否 | C++ 编译 |
| **DSR** | 深度 RL | GPU | 否 | PyTorch |
| **uDSR** | 深度 RL + poly | GPU | 否 | PyTorch |

---

## 快速开始

### 第一步：拉取代码

```powershell
git clone <your-repo-url> qwen_experiments_deps
cd qwen_experiments_deps
```

### 第二步：拉取上游代码库

```powershell
# 根据你要跑的基线，拉取对应的上游代码：
git clone https://github.com/.../In-Context-Symbolic-Regression.git external/In-Context-Symbolic-Regression
git clone https://github.com/.../LLM-SR.git external/LLM-SR
git clone https://github.com/.../SR-LLM.git external/SR-LLM
git clone https://github.com/.../GP-GOMEA.git external/GP-GOMEA
git clone https://github.com/.../LibraryAugmentedSymbolicRegression.jl.git external/LibraryAugmentedSymbolicRegression.jl
```

### 第三步：配置环境

创建 `.env` 文件（算法根目录下）：
```ini
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_API_KEY=sk-your-key-here
QWEN_MODEL=qwen3.5-plus
```

### 第四步：运行基线

每条基线有独立的虚拟环境和依赖文件。以 gplearn 为例：

```powershell
python -m venv venv-gplearn
.\venv-gplearn\Scripts\pip install -r requirements-gplearn.txt

# 小规模测试
.\venv-gplearn\Scripts\python gplearn/run_gplearn_benchmark.py --function-id 1

# 全量运行（10 个函数各 10 次独立 run）
.\venv-gplearn\Scripts\python run_all_baselines.py --baselines gplearn --runs 10
```

每条基线的完整安装与运行步骤见 [SETUP.md](./SETUP.md)。

### 第五步：收集结果

所有基线结果自动汇聚到 `results/all_baselines_summary.jsonl`：

```powershell
python run_all_baselines.py --resume
```

---

## 分机器执行策略

对比算法可分配到不同机器上独立执行，只要保证参数一致即可。

### 按 run_index 拆分

```powershell
# 机器 A：run 0-3
python run_all_baselines.py --baselines gplearn --runs 4 --base-seed 42

# 机器 B：run 4-6
python run_all_baselines.py --baselines gplearn --runs 3 --base-seed 46

# 机器 C：run 7-9
python run_all_baselines.py --baselines gplearn --runs 3 --base-seed 49
```

关键：`base_seed` 不能重叠，确保每次 run 的随机种子全局唯一。

### 按基线拆分

```powershell
# 机器 A：纯 CPU 基线
python run_all_baselines.py --baselines gplearn pysr gp_gomea

# 机器 B：LLM 基线
python run_all_baselines.py --baselines icsr llm_sr sr_llm
```

### 结果合并

各机器跑完后，将各自 `results/all_baselines_summary.jsonl` 文件拼接即可：

```powershell
Get-Content machine_A/results/all_baselines_summary.jsonl > merged_summary.jsonl
Get-Content machine_B/results/all_baselines_summary.jsonl >> merged_summary.jsonl
Get-Content machine_C/results/all_baselines_summary.jsonl >> merged_summary.jsonl
```

---

## 输出结果格式

每条运行记录包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `algorithm` | str | 算法名（gplearn, ICSR, ...） |
| `function_id` | int | 测试函数编号（1-10） |
| `run_index` | int | 独立运行编号（0-9） |
| `run_seed` | int | 本 run 的随机种子 |
| `train_mse` | float | 训练集均方误差 |
| `test_mse` | float | 测试集均方误差 |
| `runtime_seconds` | float | 运行耗时 |
| `best_expression` | str | 最优表达式 |
| `status` | str | success / failed |

---

## 参数约定

所有基线的关键参数与主实验保持一致：

| 参数 | 值 | 适用基线 |
|------|-----|---------|
| pop_size | 500 | gplearn, PySR |
| generations / niterations | 50 | gplearn, PySR |
| base_seed | 42 | 所有基线（LLM-SR 除外） |
| model | qwen3.5-plus | ICSR, LLM-SR, SR-LLM, LaSR |
| num_runs | 10 | 所有基线 |

---

## Git 管理建议

```powershell
# 初始化
git init
git add .
git commit -m "init: baseline wrappers with per-baseline requirements"

# 推送到远程
git remote add origin <your-repo-url>
git push -u origin master
```

**用 .gitignore 管理大型文件**：`external/`、`.pysr_runtime/`、`results/` 不会进入版本控制。上游代码库需在目标机器上单独克隆或同步。
