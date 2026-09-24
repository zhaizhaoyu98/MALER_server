# MALER 本地最小运行环境

_记录本机 conda 环境 `maler` 的最小安装方案与验收证据 · 2026-09-22_

---

## 📋 TL;DR

- **环境**：conda 环境 `maler`，`Python 3.7.12`，48 个包，全部锁定版本。
- **选择 3.7 的原因**：Django 2.1.8 官方仅支持 Python 3.5–3.7[^2]，且既有验证证据产生于 Python 3.7.11。
- **实测结果**：`manage.py check` 无问题，**46/46 测试通过**，5 条公开路由全部返回 200，生产口径安全审计 **10/10**。
- **两处与预期不同**：`python=3.7.11` 已从 conda-forge 下架（改用 `3.7.12`）；MRMR 必须用 `mrmr-selection`，服务器文件中记录的 `mrmr==0.9.2` 不满足代码调用。
- **本地仍无法覆盖**：外部队列与容量基准脚本依赖仓库外原始数据或 Windows 专用 API。

> 2026-09-22 更新：Windows `gene_edit` 环境使用 Python 3.7.11，并通过
> `conda run -p D:\software\miniforge3\envs\gene_edit python manage.py test`
> 当时完成 46 项测试；2026-09-24 本地 `gene_edit` 环境在患者级修正后
> 完成 50 项 Django 测试及 3 项患者分区测试。下文 `maler`/Python 3.7.12 记录保留为 Linux 最小环境的
> 独立复现证据。

---

## 🎯 目标与边界

本次安装的目标是**让项目在本机跑起来**，以支撑后续代码与稿件修订工作，因此范围界定为：

| 在范围内 | 不在范围内 |
| --- | --- |
| 启动 Django 站点、跑通 46 项测试 | 阿里云部署栈（`gunicorn`、`gevent`、Nginx） |
| 运行不依赖外部数据的验证脚本 | 重跑外部队列验证（原始矩阵未入库） |
| 复现证据环境的关键版本 | MySQL 相关包（项目实际使用 sqlite3） |

---

## 🧩 环境标识

| 项 | 值 |
| --- | --- |
| 环境名 | `maler` |
| 路径 | `/home/iseiyah/miniforge3/envs/maler` |
| 解释器 | `/home/iseiyah/miniforge3/envs/maler/bin/python` |
| Python | 3.7.12（conda-forge，build `hf930737_100_cpython`） |
| pip / setuptools / wheel | 24.0 / 67.7.2 / 0.42.0 |
| 包总数 | 48 |
| 通道 | conda-forge[^3]（仅用于安装 Python 与 pip 工具链） |

> 📌 **与证据环境的差异**：既有验证结果记录的是 Python 3.7.11，而 `3.7.11` 已不在 conda-forge 现有索引中，改用同一 3.7 分支的最后一个补丁版 `3.7.12`。两者属同一 minor 版本，不影响 Django 2.1.8 兼容性。Python 3.7 分支已于 2023-06-27 结束官方支持[^1]，因此该环境仅作复现用途，长期迁移属另一项工作。

---

## ⚙️ 安装步骤

安装按四层推进，每层完成即验证，避免一次性解析失败难以定位。

### T0 建解释器

```bash
conda install -n maler -c conda-forge --override-channels -y python=3.7.12
conda install -n maler -c conda-forge --override-channels -y "pip<24.1" "setuptools<68" wheel
```

该 Python 构建**不自带 pip**，必须单独安装；`pip` 需约束在 `24.1` 以下，因为 24.1 起要求 Python ≥ 3.8。

### T1 数值核心（锚定证据版本）

```bash
/home/iseiyah/miniforge3/envs/maler/bin/python -m pip install \
  "numpy==1.21.2" "pandas==1.3.5" "scipy==1.7.3" "scikit-learn==1.0.2" "joblib==1.3.2"
```

版本取自 `validation/results/validation_results.json` 的 `software` 块与模型包 manifest，是复现既有结果的基线。

### T2 框架

```bash
/home/iseiyah/miniforge3/envs/maler/bin/python -m pip install "django==2.1.8" "dwebsocket==0.5.12"
```

`dwebsocket` 不可省：它在 `INSTALLED_APPS` 中，且被 `mlserver/urls.py` 顶层导入的旧模块引用。

### T3 建模与出图（仅限预编译 wheel）

```bash
/home/iseiyah/miniforge3/envs/maler/bin/python -m pip install --only-binary=:all: \
  "xgboost==1.6.2" "lightgbm==3.3.3" "scikit-survival==0.17.2" \
  "osqp==0.6.7.post3" "qdldl==0.1.7.post5" lifelines matplotlib "seaborn==0.12.1"
```

`--only-binary=:all:` 用于阻止源码构建；`osqp` 与 `qdldl` 的版本必须显式钉住，原因见下节。

### T4 MRMR

```bash
/home/iseiyah/miniforge3/envs/maler/bin/python -m pip install mrmr-selection
```

### 复装一句话版本

上述五层按序执行即可复现；`conda run` 会缓冲输出，长安装建议直接调用环境内解释器路径以便观察进度。

---

## 📊 已安装清单

| 类别 | 包与版本 |
| --- | --- |
| 运行时 | `Python 3.7.12`、`pip 24.0`、`setuptools 67.7.2`、`wheel 0.42.0` |
| Web | `Django 2.1.8`、`dwebsocket 0.5.12` |
| 数值核心 | `numpy 1.21.2`、`pandas 1.3.5`、`scipy 1.7.3`、`scikit-learn 1.0.2`、`joblib 1.3.2` |
| 建模 | `xgboost 1.6.2`、`lightgbm 3.3.3`、`scikit-survival 0.17.2`、`lifelines 0.27.8`、`mrmr-selection 0.2.8` |
| 出图 | `matplotlib 3.5.3`、`seaborn 0.12.1` |
| 传递依赖 | `osqp 0.6.7.post3`、`qdldl 0.1.7.post5`、`ecos 2.0.14`、`numexpr 2.8.6`、`statsmodels 0.13.5`、`polars 0.18.4`、`category-encoders 2.6.4`、`formulaic 1.1.1`、`autograd 1.6.2`、`autograd-gamma 0.4.2`、`patsy 1.0.3`、`pytz 2026.3.post1` 等 |

<details>
<summary><strong>📋 完整 pip freeze（48 行）</strong></summary>

```text
astor==0.8.1
autograd==1.6.2
autograd-gamma==0.4.2
cached-property==1.5.2
category-encoders==2.6.4
cycler==0.11.0
Django==2.1.8
dwebsocket==0.5.12
ecos==2.0.14
fonttools==4.38.0
formulaic==1.1.1
future==1.0.0
graphlib_backport==1.1.0
importlib-metadata==6.7.0
importlib-resources==5.12.0
interface-meta==1.3.0
Jinja2==3.1.6
joblib==1.3.2
kiwisolver==1.4.5
lifelines==0.27.8
lightgbm==3.3.3
MarkupSafe==2.1.5
matplotlib==3.5.3
mrmr-selection==0.2.8
numexpr==2.8.6
numpy==1.21.2
osqp==0.6.7.post3
packaging==24.0
pandas==1.3.5
patsy==1.0.3
Pillow==9.5.0
polars==0.18.4
pyparsing==3.1.4
python-dateutil==2.9.0.post0
pytz==2026.3.post1
qdldl==0.1.7.post5
scikit-learn==1.0.2
scikit-survival==0.17.2
scipy==1.7.3
seaborn==0.12.1
six==1.17.0
statsmodels==0.13.5
threadpoolctl==3.1.0
tqdm==4.68.2
typing_extensions==4.7.1
wrapt==1.16.0
xgboost==1.6.2
zipp==3.15.0
```

</details>

---

## 🔍 安装过程中的关键发现

### 1. `scikit-survival` 的传递依赖会触发源码构建

`scikit-survival 0.17.2` 依赖 `osqp`，而 `osqp` 依赖 `qdldl`。pip 默认选择的 `qdldl 0.1.9.post1` **没有 Python 3.7 的预编译 wheel**，会退化为源码构建并因缺少 CMake 与 C 编译器而失败[^5]。锁定的 `qdldl 0.1.7.post5` 提供 `cp37` wheel，可完全避开编译。

同理，`scikit-survival 0.17.2` 是 Python 3.7 上最后一个提供 `cp37` wheel 的版本[^6]，更高版本在 3.7 上只能源码构建，不适合最小环境。

### 2. MRMR 的正确发行包是 `mrmr-selection`

| 项 | 服务器文件记录 | 实际实现要求 | 结论 |
| --- | --- | --- | --- |
| 发行包 | `mrmr==0.9.2` | `mrmr-selection` | 装 `mrmr-selection==0.2.8`[^4] |
| 依据 | `ml_python37.yml` 的 pip 段 | `mlserver/safe_ml.py:169` 报错文案，且 `safe_ml.py:163-167` 调用时传 `show_progress`、`n_jobs` | 探针实测 |

实测签名确认 `mrmr_classif(X, y, K, ..., n_jobs=-1, show_progress=True)` 接受代码所传的两个参数，与 `mrmr-selection` 一致：

```text
mrmr_classif : (X, y, K, relevance='f', redundancy='c', denominator='mean', cat_features=None,
                cat_encoding='leave_one_out', only_same_domain=False, return_scores=False,
                n_jobs=-1, show_progress=True)
```

> 📌 该结论同时是论文 Table 1 与 Methods 中 MRMR 描述的事实依据，需与稿件措辞对齐。

### 3. 启动链强制加载整套科学栈

`mlserver/urls.py` 在顶层导入 6 个旧 websocket 结果模块（路由已注释）。这些模块在模块级导入 `xgboost`、`lightgbm`、`scikit-survival`、`lifelines`、`matplotlib`、`seaborn`、`dwebsocket`，因此**即便只想打开首页，这些包也必须在位**。这也是本次「最小安装」仍达 48 个包的原因。

---

## ✅ 验收证据

| 检查 | 命令 | 结果 |
| --- | --- | --- |
| Django 系统检查 | `manage.py check` | 无问题（0 silenced） |
| 导入探针 | 13 个第三方包 + `django.setup()` | 全部成功，版本与上表一致 |
| 自动化测试 | `manage.py test mlserver -v 1` | **Ran 46 tests — OK** |
| 安全审计（本地口径） | `manage.py audit_security_configuration --strict` | 4/10（缺 TLS/HSTS 类设置，属预期） |
| 安全审计（生产口径） | 同上，附 `DJANGO_SECURE_SSL=true`、`DJANGO_HSTS_SECONDS=31536000`、强密钥 | **10/10** |
| 路由冒烟 | `runserver 127.0.0.1:8971` + `curl` | `/maler/home`、`/maler/analysis`、`/maler/analysis/methods.json`、`/maler/predict`、`/maler/help` 全部 200 |
| 方法注册表 | 同上 `methods.json` | 暴露 `registry_version`、`models`、`feature_reduction` |

`feature_reduction` 实测返回：`none`、`select_k_best`、`pca`、`mrmr`、`fss`、`bss`，与稿件中候选降维范围的描述一致。

**`check --deploy` 的 5 条警告**（HSTS、SSL 重定向、`SECRET_KEY`、`SESSION_COOKIE_SECURE`、`CSRF_COOKIE_SECURE`）是**预期的本地状态**：这些设置由环境变量驱动，生产部署时才开启。

---

## 🚫 本地无法覆盖的部分

| 脚本 | 原因 |
| --- | --- |
| `run_external_gse37745.py`、`audit_external_data_quality.py` | 需 `external_data/E-GEOD-37745.processed.1.zip` 与 `.sdrf.txt`，仓库内只有元数据 JSON |
| `run_external_gse50081.py` | 需 GSE50081 series matrix，未入库 |
| `run_survival_expanded_validation.py` | 需 TCGA / CGGA 矩阵；GBSG2 可由 `lifelines` 内置数据集加载 |
| `benchmark_capacity.py` | 使用 `ctypes.wintypes` 与 `PROCESS_MEMORY_COUNTERS_EX` 读取峰值工作集，属 Windows 专用 API |

其余本地脚本（`run_reviewer_validation.py`、`run_pca_sensitivity.py`、`run_equal_budget_baselines.py`、`audit_survival_data_quality.py`、`audit_legacy_survival_model.py`、`audit_production_security.py`）可使用仓库内 `mlserver/static/cache/example/` 的示例数据运行。

> ⚠️ **运行前请注意**：这些脚本会直接覆盖写 `validation/results/*.json`。既有证据产生于 Windows + MKL 环境，在本机重跑可能出现细微数值差异；建议在分支或独立副本中运行，避免污染已提交证据。

---

## 🧹 仓库遗留问题

| 问题 | 影响 | 建议 |
| --- | --- | --- |
| `mlserver/migrations/__pycache__/__init__.cpython-37.pyc` 被 git 跟踪 | 每次运行测试或服务都会改写该文件，制造假改动 | 在可复现打包阶段从索引移除并加入忽略规则 |
| `temp/` 已被 `.gitignore` 忽略 | 放在其中的本地笔记不会同步给协作者 | 需共享的内容放到 `docs/` |

---

## 🔗 参考

- [`deployment/README.md`](../deployment/README.md) — 生产部署顺序与检查项
- [`validation/README.md`](../validation/README.md) — 评估设计与证据口径
- [`ml_python37.yml`](../ml_python37.yml) — Windows 服务器环境导出（win-64，不能用于 Linux）
- [`revision-status.md`](revision-status.md) — 返修工作状态与执行计划

[^1]: Python Developer's Guide. "Status of Python versions"（Python 3.7 已于 2023-06-27 结束支持）https://devguide.python.org/versions/
[^2]: Django Documentation. "Django 2.1 release notes"（支持 Python 3.5–3.7）https://docs.djangoproject.com/en/2.1/releases/2.1/
[^3]: conda-forge. "The conda-forge organization" https://conda-forge.org/
[^4]: PyPI. "mrmr-selection" https://pypi.org/project/mrmr-selection/
[^5]: PyPI. "qdldl"（0.1.9.post1 无 cp37 wheel，0.1.7.post5 有）https://pypi.org/project/qdldl/
[^6]: PyPI. "scikit-survival 0.17.2"（Python 3.7 可用的最后一个带 cp37 wheel 的版本）https://pypi.org/project/scikit-survival/0.17.2/
