# MALER 回复信最终一致性记录

_用于核对 `v1.0.0-review2` 回复信、正文、补充材料和代码是否一致_

## 已完成的关键修正

| 原风险 | 最终回复口径 |
| --- | --- |
| lifelines CoxPHFitter 与 p 值排序 | 使用自实现的 censoring-aware vectorized Cox score statistic |
| 300 MB 上传上限 | 数据 200 MB，模型 100 MB |
| mean imputation | 训练折内 median imputation |
| 非折内或非 nested | 外层重复验证、内层选择；所有拟合步骤限制在训练折 |
| 已完成功能仍写 future | PCA、no reduction、扩展指标、完整 bundle、bootstrap CI 改为已实现 |
| legacy 路径“不可访问”但仍有 helper | `get_cp_combination` 已取消路由，公共计算统一走验证引擎 |
| 36 项测试 | 更新为 46 项测试全部通过 |
| 仓库 pending | 更新为 GitHub/Gitee 公开、MIT、`v1.0.0-review2` |
| Figure 2 六面板 | 更新为含 scaling guidance 的七面板正式合图 |
| Figure 3 旧页面截图 | 更新为 checksum-locked 定量验证图 |
| 预测对齐只写“允许重排” | 增加页面报告重排列和忽略额外特征的说明 |

## 可直接使用的公开仓库声明

Source code and reproducibility materials are publicly available under the MIT
License at <https://github.com/zhaizhaoyu98/MALER_server>, mirrored at
<https://gitee.com/zhaoyuzhai/MLSERVER>, as release `v1.0.0-review2`.
The release includes pinned Windows and Linux Conda specifications, example
data, validation and figure-generation scripts, fixed seeds, source hashes,
machine-readable results, deployment templates, and 46 passing Django tests.

不要声称存在 notebook；审稿意见允许 reproducible scripts or notebooks，当前发布采用
脚本路径。

## 云端声明边界

可写：2026-09-22 对 `13d6b3e` 的阿里云 HTTP 部署完成主页、分析、预测、帮助和
方法注册表冒烟检查；缓存静态路径返回 404；六项非 TLS 应用控制通过。

不可写：review2 已部署、HTTPS 已完成、HSTS/secure-cookie 已实机通过、独立渗透
测试完成、存储已加密、备份恢复已验证。

本轮用户明确要求不更新阿里云，因此回复信必须区分“公开源码 review2”和
“线上已验证版本 13d6b3e”。

## 预测特征对齐回复

The signed model manifest stores the ordered feature contract. External
matrices are aligned by identifier rather than position; missing or duplicated
required identifiers are rejected. Safe reordering is allowed, and the result
page reports both order differences and unexpected columns that were ignored.
The fitted preprocessing pipeline is reused unchanged, so these notices do not
alter prediction values.

## 图表回复

- Figure 1：简化的 leakage-controlled 方法流程。
- Figure 2：七面板界面图，新增 scaling guidance，并与 A-G 图注逐项对应。
- Figure 3：312 个开发样本、50 个外层验证估计、302 个 held-out 样本、ROC/PR、
  混淆矩阵及 bootstrap 置信区间。
- Figure S7：与 Figure 2 使用相同的七个经核验界面面板。
- Table 3：结构化比较 MALER、Auto-WEKA、auto-sklearn、iLearnPlus 和 JADBio，
  不声称普遍性能优越。

## 保持为 future 的内容

校准曲线、高级插补、全面 AutoML 定量基准、正式用户研究、容器化发行、开放交换
格式、受支持版本迁移，以及前瞻性临床/实验验证仍保持 future 或 limitation。

## 作者仍需人工确认

- Word/Zotero Refresh 不出现缺失条目。
- 作者、单位、基金、通讯作者、利益冲突和伦理声明真实无误。
- 投稿系统中的文件类别和期刊图片规范。
- 服务器维护期限与负责人。

最终回复信应保持 55 个 Comment、55 个 Response 和 55 个
Changes in the manuscript 段落，且每项完成声明均能由代码、公开仓库、图或验证记录
直接支持。
