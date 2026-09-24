# MALER 审稿意见与实现侧最终对照

_2026-09-24 患者级复核版；当前公开版本为 `v1.0.0-review4`，`v1.0.0-review3` 为旧基线_

## 结论

公开计算统一走验证引擎；TCGA 风格样本条形码还会经过患者级分区审计，
跨训练/测试分区的患者两侧均排除，同分区多份样本按样本类型和 vial 的预设
规则保留一份。原始 CSV 不改动；具体排除清单写入结果 JSON。
患者级修正已作为 `v1.0.0-review4` 发布；此前 `v1.0.0-review3` 不包含此修正。

## 已实现事项

| 事项 | 实现证据 | 投稿表述 |
| --- | --- | --- |
| 折内处理与嵌套验证 | `safe_ml.py`、`validated_analysis.py` | 插补、缩放、特征处理和调参均在训练折内拟合 |
| 任务优化指标 | `model_registry.py` | 分类 balanced accuracy、回归 R²、生存 C-index |
| 26 个模型注册信息 | `model_registry.py`、`analysis/methods.json` | 类名、默认值、网格、缩放要求可机器读取 |
| 输入质控 | `safe_ml.py` | 拒绝空值结构、重复标识、非数值、无穷及不合格结局 |
| 标识符预测对齐 | `align_prediction_frame` | 不依赖输入位置 |
| 对齐偏差提示 | `alignment_report`、`predict_result.html` | 报告重排列与忽略的额外特征 |
| 完整模型导出 | `.maler` manifest、签名和 Pipeline | 不是裸估计器或通用交换格式 |
| PCA 与 no reduction | 验证引擎、结果 JSON | 可选分支和同重采样对照；候选数不等，不称等计算预算消融 |
| 患者级分区审计 | `patient_partitions.py`、`validated_analysis.py` | binary 309/299、multiclass 482/468、regression 367/162、survival 160/133 |
| 扩展指标 | 结果 CSV/JSON | specificity、MCC、PR-AUC、time-dependent AUC、IBS |
| 外部与跨队列验证 | `validation/results` | 如实报告正面与不理想结果 |
| 访问与缓存控制 | `task_access.py`、cleanup command | 任务令牌、删除入口和保留期配置 |
| legacy 路由收敛 | `mlserver/urls.py` | 旧 WebSocket 和 `get_cp_combination` 不再提供计算路径 |
| 自动测试 | `mlserver/tests.py` | 50/50 通过，另有 3 项患者分区单元测试 |
| UI 鼠标回归 | Analysis、Predict、Preview、Help、结果页 | 此前版本 27/27 通过；患者级提示新增后以 Django 测试与本轮页面复核为准 |
| 签名模型闭环 | `.maler` 下载与 Predict 上传 | 本地测试密钥下完成生成、下载、上传和 50 样本预测 |

## 仓库物料

| 物料 | 状态 |
| --- | --- |
| README | 已完成，含安装、启动、验证与安全边界 |
| LICENSE | MIT |
| Windows 环境 | `environment_windows.yml` |
| Linux 环境 | `environment_server.yml` |
| 示例数据 | 四类任务示例位于静态示例目录 |
| 验证脚本和结果 | `validation/` 与 `validation/results/` |
| 公开地址 | GitHub 与 Gitee 均可匿名读取 |
| 版本标签 | `v1.0.0-review4` 包含患者级修正；`v1.0.0-review3` 为旧公开基线 |

审稿意见使用“scripts or notebooks”。现有脚本、示例、固定随机种子、校验哈希和
机器可读结果已经满足可复现路径，因此回复信不声称提供 notebook。

## 已完成的文字一致性修正

- `mean imputation` 改为 fold-local median imputation。
- 上传上限固定为 200 MB。
- “非折内、非 nested”改为当前嵌套验证事实。
- Cox p-value/CoxPHFitter 描述改为 censoring-aware Cox score statistic。
- PCA、no-reduction、扩展指标、完整 pipeline 导出和 bootstrap 区间改为已实现。
- 旧计算模块改为无公共计算路由，不再称仍可替代运行。
- 测试数量更新为 50 项 Django 测试；患者级脚本另有 3 项单元测试。
- 正文需区分网页默认 5 折×2 重复与论文脚本 5 折×10 重复。
- 内部留出集使用 1,000 次 bootstrap，外部队列和扩展生存使用 2,000 次。
- GSE50081 在此前版本已报告，本轮仅称回顾性外部重分析，不称全新未触碰测试。
- 缓存访问令牌 24 小时失效；物理删除依赖运营方执行清理命令，不宣称已验证定时任务。
- 预测对齐回复增加重排列与额外特征页面提示。
- 修复 Survival 切回其他任务后 ANOVA/MRMR 仍被禁用的问题。
- 修复 Analysis 横向溢出和 Preview 固定宽度图表越界，并移除无效第三方脚本。

## 明确保留为 future 的事项

- 生存校准曲线。
- 高级或多重插补。
- 用户可配置的全部 TopK/候选池范围。
- 容器化本地发行包。
- 系统性跨 AutoML 定量排名。
- PMML/ONNX/WDL 导出。
- 正式参与者可用性研究。
- 前瞻性临床与实验验证。
- Python 3.7/Django 2.1 迁移。

## 工程卫生后续项

以下不是本地患者级修订的完成条件，但后续应处理：旧 `.bak` 模板、
`send_email.py` 的其余未使用历史代码、定时缓存清理、部署维护责任和更多估计器的独立参考对照。
二分类 Logistic 与回归 Ridge 已由 `validation/reference_parity_check.py` 直接使用 scikit-learn 复核。
本轮已移除该旧模块在导入时使用写死地址发信的语句；公共网页通知路径仍按 SMTP 配置条件运行。

相关状态见 [`revision-status.md`](revision-status.md)，最终回复口径见
[`response-letter-changes.md`](response-letter-changes.md)。
