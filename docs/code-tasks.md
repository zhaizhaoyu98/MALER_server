# MALER 审稿意见与实现侧最终对照

_按 `v1.0.0-review2` 代码和 46 项测试更新_

## 结论

审稿意见涉及的本轮 P0 代码任务已经完成。公开计算统一走验证引擎；预测矩阵
对齐偏差会显示给用户；最后一个 legacy helper 路由已取消。当前没有为迎合文字
而回退实现的事项，正文与回复信以代码事实为准。

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
| PCA 与 no reduction | 验证引擎、结果 JSON | 可选分支和等预算基线均已实现 |
| 扩展指标 | 结果 CSV/JSON | specificity、MCC、PR-AUC、time-dependent AUC、IBS |
| 外部与跨队列验证 | `validation/results` | 如实报告正面与不理想结果 |
| 访问与缓存控制 | `task_access.py`、cleanup command | 任务令牌、删除入口和保留期配置 |
| legacy 路由收敛 | `mlserver/urls.py` | 旧 WebSocket 和 `get_cp_combination` 不再提供计算路径 |
| 自动测试 | `mlserver/tests.py` | 46/46 通过 |

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
| 版本标签 | `v1.0.0-review2` |

审稿意见使用“scripts or notebooks”。现有脚本、示例、固定随机种子、校验哈希和
机器可读结果已经满足可复现路径，因此回复信不声称提供 notebook。

## 已完成的文字一致性修正

- `mean imputation` 改为 fold-local median imputation。
- 上传上限固定为 200 MB。
- “非折内、非 nested”改为当前嵌套验证事实。
- Cox p-value/CoxPHFitter 描述改为 censoring-aware Cox score statistic。
- PCA、no-reduction、扩展指标、完整 pipeline 导出和 bootstrap 区间改为已实现。
- 旧计算模块改为无公共计算路由，不再称仍可替代运行。
- 测试数量更新为 46。
- 预测对齐回复增加重排列与额外特征页面提示。

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

以下不阻塞 review2，但后续应处理：旧 `.bak` 模板、孤立 `send_email.py`、
定时缓存清理、部署维护责任和独立参考脚本对照。

相关状态见 [`revision-status.md`](revision-status.md)，最终回复口径见
[`response-letter-changes.md`](response-letter-changes.md)。
