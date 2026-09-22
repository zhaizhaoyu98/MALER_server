# MALER 回复信待改清单

_逐条列出 `MALER_Response_Letter_Draft.docx` 中必须修改、可填实与应保持原样的内容 · 依据本地代码实测 · 2026-09-22_

---

## 📋 TL;DR

- 回复信草稿**整体可用**，但有 **6 处文字与当前代码不符**，其中 3 处会被审稿人对照代码直接证伪，优先处理。
- 草稿中 **9 处 `AUTHOR VERIFICATION REQUIRED`** 里，**7 处可由本地代码直接填实**（仅 2 处属服务器侧，须在线上核实）。
- 草稿把 6 类**已经实现**的功能写成了 future version，1 类（校准曲线）确实仍未实现，不要误改。
- 本轮完成的 E1（预测特征对齐提示）意味着 R3 Minor 9 的措辞需从 "planned for a future version" 升级为已实现。

---

## 📄 必须修改（文字与代码不符）

| # | 位置 | 草稿原文立场 | 应改为 | 依据 |
| --- | --- | --- | --- | --- |
| C1 | R3 Major 4 | "implemented with lifelines `CoxPHFitter` … prioritises features by univariate Cox p-value" | 单因素 Cox 排序由本仓库自实现的**向量化 Cox score statistic** 完成（`mlserver/safe_survival.py` 的 `CoxPHSelectKBest`），不依赖 lifelines，也不按 p 值排序 | `safe_survival.py:80-141`；lifelines 仅存在于 legacy 模块 |
| C2 | R1#6 | "limits individual files to 300 MB" | 上限为 **200 MB** | `ML_WebServer/settings.py:93`；`deployment/nginx_maler.conf.example` 同为 200m |
| C3 | R1#10、R3 Major 7 | "simple mean imputation" | 缺失值以**训练折内中位数**填补（`SimpleImputer(strategy="median")`），并随 Pipeline 一起导出 | `mlserver/safe_ml.py:262-281` |
| C4 | R2#5、R3 Major 2 | "the present workflow is not a fully nested or fully fold-isolated pipeline" 并列为 future | 当前实现**在折内重拟合**插补、缩放、降维与调参，并使用嵌套外层/内层循环；应改为描述现状，同时说明网页默认 5×2×3 与论文分析设计 5×10×3 的区别 | `safe_ml.py:445-479,506-537`；`validated_analysis.py:238-253` |
| C5 | R1#3、R1#5、R2#2、R2#8、R2#9、R3 Major 1、R3 Minor 3、R3 Minor 8 | 把 PCA、扩展分类指标、扩展生存指标、多任务验证、完整 pipeline 导出、留出集置信区间列为 future | 见下方"应改写为已实现"一节 | 见该节逐条依据 |
| C6 | R3 Major 3、补充材料 | 旧 WebSocket 模块被描述为"不可访问的兼容性代码" | 路由确已禁用，但 6 个模块仍被 `mlserver/urls.py` 顶层导入，且 `/maler/get_cp_combination` **仍指向 legacy 模块** | `mlserver/urls.py:5-8,69` |

### 应改写为已实现（C5 展开）

| 事项 | 草稿位置 | 可写入的事实 |
| --- | --- | --- |
| PCA / 特征提取 | R1#3 | 已作为可选降维（UI 选项 + `methods.json` 的 `feature_reduction`），并明确说明返回成分而非基因签名 |
| specificity / balanced accuracy / MCC / PR-AUC / 多分类 macro 指标 | R1#5、R2#8 | 均已计算并在结果页展示（`internal_cv_summary.csv`；`validated_result.html` 的通用指标表） |
| time-dependent AUC、integrated Brier score | R2#9、R3 Major 5 | 均已实现；**仅校准曲线仍缺失**，只有这一项可保留 future 表述 |
| 多分类 / 回归 / 生存的验证与无选择基线 | R2#2、R3 Major 1 | 四类任务与等预算无选择基线均已产出结果文件 |
| 完整 pipeline 导出 | R3 Minor 8 | 导出物为完整 Pipeline + `manifest.json`（含特征顺序、类名、六个包版本）+ 签名 |
| 留出集置信区间 | R3 Minor 3 | 已提供 2000 次 bootstrap 区间（`heldout_test_summary.csv`） |

---

## ✏️ 因本轮实现而需升级的表述

| 位置 | 现状措辞 | 建议改为 |
| --- | --- | --- |
| R3 Minor 9 / R2#19 | "stronger automated detection and more informative warnings for incompatible external matrices are planned for a future version" | 预测矩阵现在**会报告**特征顺序不一致与被忽略的多余列（结果页面显示提示）；仍不提供"更强的检测"，该部分可继续列为 future |
| 对应正文方法部分 | 未描述对齐提示 | 增加一句：外部预测矩阵按标识对齐，若列顺序与训练契约不同或含多余列，页面会提示且预测不受影响 |

> 📌 若最终决定不在稿件中提及该提示，则回复信不要声称已实现，避免再次产生文字与代码的落差。

---

## ✅ 可填实 `AUTHOR VERIFICATION REQUIRED`（本地已可回答）

| # | 标记内容 | 可直接写入回复信的事实 | 依据 |
| --- | --- | --- | --- |
| 1 | 网格搜索的评分标准与 CV 切分器（标记 #44） | 分类按 balanced accuracy、回归按 R²、生存按 concordance index 调参；切分器为 `RepeatedStratifiedKFold`（分类）与 `RepeatedKFold`（回归），随机种子 10；全部参数、默认值与搜索范围见机器可读注册表 | `model_registry.py:20,86,146`；`analysis/methods.json`；UI `analysis.html:2374-2375` |
| 2 | 导出模型包含什么（标记 #73、#120） | 导出的是**完整 Pipeline**（插补→缩放→降维→估计器）而非裸估计器；`manifest.json` 记录特征名与顺序、`model_class`、六个依赖包版本；另有 `signature.sha256` 签名 | `safe_ml.py:612-649` |
| 3 | 哪些输入质控已实现（标记 #114） | 空矩阵、样本标识重复、特征标识重复、非数值、无穷值、全缺失、超 80% 缺失、零方差，均会拒绝并给出可读错误 | `safe_ml.py:73-118` |
| 4 | 生存状态的接受值与边界（标记 #117） | 接受布尔；数值仅 `0/1`（1=事件，0=截尾）；字符串不区分大小写（去空格后）接受 `1`、`1.0`、`event`、`dead`、`deceased`、`death` 与 `0`、`0.0`、`censored`、`alive`；随访时间必须有限且 > 0；另要求至少 2 例事件与 1 例截尾 | `safe_survival.py:27-64` |
| 5 | 数据存储与保留策略（标记 #61，代码侧部分） | 上传数据与结果位于服务器缓存目录的任务子目录；默认保留 24 小时后由 `cleanup_maler_cache` 清理；结果与模型下载受任务能力令牌保护；提供任务删除入口；邮件地址仅用于发送完成通知，不写入结果文件 | `settings.py:96-99`；`management/commands/cleanup_maler_cache.py`；`task_access.py`；`validated_analysis_views.py:56-70` |
| 6 | 安全机制（标记 #64、#98，代码侧部分） | 结果与模型链接需能力令牌，公开分析流程不依赖用户账号；上传上限 200 MB；不存在"已加密存储"的声明 | 同上 |
| 7 | 训练/测试划分细节（标记 #85） | 划分来自示例数据自带的 `set` 标记，代码只做校验（train/test 必须都存在且覆盖全部样本），不做随机划分 | `validation/run_reviewer_validation.py:87-125` |

**仍需线上核实（不要在本地凭空填写）**

| # | 标记内容 | 为什么不能本地回答 |
| --- | --- | --- |
| 8 | HTTPS/TLS、证书链、静态存储加密（标记 #64、#98） | 属阿里云部署状态，按约定部署在最后一次性执行 |
| 9 | 备份策略、管理员访问范围、依赖更新周期、服务维护期（标记 #61、#98） | 服务器运维事实，需在线上确认 |

---

## 📦 补齐物料后需追加的声明

| 若完成 | 则回复信可写 | 否则必须保持的措辞 |
| --- | --- | --- |
| `LICENSE` | 明确写出许可证类型 | 不声称有开源许可证 |
| Linux 环境文件 + 根 README | "仓库提供可复现的依赖清单与运行步骤" | 只描述为"已提供 Windows 服务器环境导出" |
| 参考 notebook / 教程材料 | 保留现有"public repository containing example data/tutorial material and reference notebooks" | 必须核实这些 notebook 是否在别的公开仓库；若不存在则删除该承诺 |

> ⚠️ 当前仓库内 `.ipynb` 数量为 **0**、无 `LICENSE`，而草稿的 Data Availability 与 R2#10、R3 Major 8 已引用这些材料。

---

## ⏸️ 应保持 future 的条目（不要误改为已实现）

| 事项 | 草稿位置 |
| --- | --- |
| 校准曲线（calibration plot） | R2#9、R3 Major 5 |
| 高级插补方法 | R1#10、R3 Major 7 |
| 算法特定的预处理管道 | R3 Major 7 |
| Top-50 / TopK 的用户可配置范围 | R2#7 |
| 容器化本地部署 | R2#12、R2#13 |
| 系统性小/中/大规模性能基准 | R2#11 |
| 与独立 Python/R 脚本的同划分基准 | R2#10 |
| WDL / PMML / ONNX 工作流导出 | R1#11 |
| 正式可用性用户研究 | R1#8 |

---

## 🔗 相关文档

- [`code-tasks.md`](code-tasks.md) — 审稿意见与实现的完整对照
- [`environment-setup.md`](environment-setup.md) — 本地环境事实（版本清单可用于可复现性声明）
- [`revision-status.md`](revision-status.md) — 返修整体状态与执行计划
