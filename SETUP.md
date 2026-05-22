# 对比算法独立运行指南

每个对比算法有独立的虚拟环境和 requirements.txt，可单独部署到任意机器上运行。

## 准备工作（每台机器只需一次）

```powershell
# 1. 复制项目到目标机器
# 需要复制的目录：
#   qwen_experiments_deps/          # 主代码 + 配置文件 + 数据集
#   qwen_experiments_deps/external/  # 上游代码库（ICSR, LLM-SR, SR-LLM, GP-GOMEA 等）
#
# 2. 配置 .env 文件（在 qwen_experiments_deps/ 下创建）
```

`.env` 文件内容：
```ini
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_API_KEY=sk-your-key-here
QWEN_MODEL=qwen3.5-plus
```

## 分基线安装与运行

### gplearn（零 token 消耗，纯 CPU）

```powershell
python -m venv venv-gplearn
.\venv-gplearn\Scripts\pip install -r requirements-gplearn.txt

# 小规模测试
.\venv-gplearn\Scripts\python gplearn/run_gplearn_benchmark.py --function-id 1

# 全量运行（10 函数 × 10 次）
.\venv-gplearn\Scripts\python run_all_baselines.py --baselines gplearn --runs 10
# 或直接
.\venv-gplearn\Scripts\python gplearn/run_gplearn_benchmark.py --all
```

### ICSR（LLM 类，需 API key）

```powershell
python -m venv venv-icsr
.\venv-icsr\Scripts\pip install -r requirements-icsr.txt
.\venv-icsr\Scripts\pip install -r external/In-Context-Symbolic-Regression/requirements.txt

# 小规模测试
.\venv-icsr\Scripts\python wrappers/icsr/run.py --function-id 1

# 全量运行
.\venv-icsr\Scripts\python wrappers/icsr/run.py --all
```

### LLM-SR（LLM 类，需 API key）

```powershell
python -m venv venv-llm_sr
.\venv-llm_sr\Scripts\pip install -r requirements-llm_sr.txt
.\venv-llm_sr\Scripts\pip install -r external/LLM-SR/requirements.txt

# 小规模测试
.\venv-llm_sr\Scripts\python wrappers/llm_sr/run.py --function-id 1

# 全量运行
.\venv-llm_sr\Scripts\python wrappers/llm_sr/run.py --all
```

### SR-LLM（LLM 类，需 PyTorch + API key）

```powershell
python -m venv venv-sr_llm
.\venv-sr_llm\Scripts\pip install -r requirements-sr_llm.txt
# 上游依赖较多，推荐用 conda 导入 external/SR-LLM/environment.yml

# 小规模测试
.\venv-sr_llm\Scripts\python wrappers/sr_llm/run.py --function-id 1

# 全量运行
.\venv-sr_llm\Scripts\python wrappers/sr_llm/run.py --all
```

### PySR（需 Julia 运行时）

```powershell
python -m venv venv-pysr
.\venv-pysr\Scripts\pip install -r requirements-pysr.txt
# PySR 会自动下载 Julia（首次运行 bootstrap 约几分钟）

# 小规模测试
.\venv-pysr\Scripts\python wrappers/pysr/run.py --function-id 1

# 全量运行
.\venv-pysr\Scripts\python wrappers/pysr/run.py --all
```

### LaSR（需 Julia + API key）

```powershell
python -m venv venv-lasr
.\venv-lasr\Scripts\pip install -r requirements-lasr.txt
# 确保 PySR 的 Julia 环境已就绪（见上面 PySR 步骤）
# 然后初始化 LaSR 的 Julia 环境：
cd external/LibraryAugmentedSymbolicRegression.jl
julia --project=. -e "using Pkg; Pkg.instantiate()"
cd ../..

# 小规模测试
.\venv-lasr\Scripts\python wrappers/lasr/run.py --function-id 1

# 全量运行
.\venv-lasr\Scripts\python wrappers/lasr/run.py --all
```

### GP-GOMEA（需编译 C++ 扩展）

```powershell
python -m venv venv-gp_gomea
.\venv-gp_gomea\Scripts\pip install -r requirements-gp_gomea.txt

# 编译 C++ 扩展（需 Visual Studio Build Tools）：
cd external/GP-GOMEA/pythonpkg
python setup.py build_ext --inplace
cd ..\..

# 小规模测试
.\venv-gp_gomea\Scripts\python wrappers/gp_gomea/run.py --function-id 1

# 全量运行
.\venv-gp_gomea\Scripts\python wrappers/gp_gomea/run.py --all
```

## 结果对比

所有基线的输出使用统一的 schema（`utils/baseline_result_schema.py`），输出到各自 `OUTPUT_PATH` 下的 `run_summary.json`。最终汇总到 `results/all_baselines_summary.jsonl`。

可以用统一脚本收集已完成的基线结果：
```powershell
.\venv-gplearn\Scripts\python run_all_baselines.py --resume
```

跨机器结果合并：把各机器的 `results/all_baselines_summary.jsonl` 文件拼接即可。
