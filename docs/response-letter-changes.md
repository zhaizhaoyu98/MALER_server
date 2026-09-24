# MALER 回复信最终一致性记录

_2026-09-24 更新：核对本地 `final_submission_patient_level_20260923`；患者级修正对应 `v1.0.0-review4`_

## 已完成的关键修正

| 原风险 | 最终回复口径 |
| --- | --- |
| lifelines CoxPHFitter 与 p 值排序 | 使用自实现的 censoring-aware vectorized Cox score statistic |
| 300 MB 上传上限 | 数据 200 MB，模型 100 MB |
| mean imputation | 训练折内 median imputation |
| 非折内或非 nested | 外层重复验证、内层选择；所有拟合步骤限制在训练折 |
| 已完成功能仍写 future | PCA、no reduction、扩展指标、完整 bundle、bootstrap CI 改为已实现 |
| legacy 路径“不可访问”但仍有 helper | `get_cp_combination` 已取消路由，公共计算统一走验证引擎 |
| 36 项测试 | 更新为 50 项 Django 测试及 3 项患者分区测试全部通过 |
| 仓库版本 | GitHub/Gitee 公开 `v1.0.0-review4`、MIT；旧 `v1.0.0-review3` 不含患者级修正 |
| Figure 2 六面板 | 更新为含 scaling guidance 的七面板正式合图 |
| Figure 3 旧页面截图 | 更新为 checksum-locked 定量验证图 |
| 预测对齐只写“允许重排” | 增加页面报告重排列和忽略额外特征的说明 |

## 公开仓库声明边界

旧 `v1.0.0-review3` 可用于追溯此前公开基线，但不能证明患者级重算。
本轮源码、机器可读结果、生成脚本与投稿图件已发布为 `v1.0.0-review4`；
正文和回复信引用该标签。阿里云应用源码于 2026-09-24 更新到 `35aa116`；
该提交晚于发布标签，但包含同一患者级修正。终稿上传时仍应核对公开链接。

本轮补充：R2#10 现在有两项直接 scikit-learn 参考拟合；R2#14–16 的回复区分
令牌过期、运营方物理清理、主机访问与未核实的备份/邮件上游日志；Editorial Major4
改为准确说明生存 FSS/BSS 与固定 k 停止；Editorial Major6–7 增加类别失衡警告、
缩放可覆盖的明确说明。上述源码现已部署；不把页面级冒烟检查写成端到端上传验证。

不要声称存在 notebook；审稿意见允许 reproducible scripts or notebooks，当前发布采用
脚本路径。

## 云端声明边界

可写：2026-09-22 对旧版 `13d6b3e` 的 HTTP 部署完成应用控制检查；
2026-09-24 将患者级修正版 `35aa116` 同步至 Aliyun 并重载 workers，先在独立
服务器工作树通过 50+3 测试，再确认公网主页、分析、预测和帮助页返回 HTTP 200，
两类上传页可见授权确认控件。

不可写：HTTPS 已完成、HSTS/secure-cookie 已实机通过、独立渗透测试完成、
存储已加密、备份恢复已验证、生产清理已定时运行或线上上传/删除已经端到端演练。

回复信区分“公开源码和结果标签 `v1.0.0-review4`”、“本地投稿 DOCX”、
“当前线上应用源码 `35aa116`”与仍然适用的 HTTP-only 安全边界。

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
- Figure 3：309 位开发患者、50 个外层验证估计、299 位内部留出患者、ROC/PR、
  混淆矩阵及 bootstrap 置信区间。
- Supplementary Figures S1–S6：分别对应内部验证、外部验证、特征稳定性、
  扩展生存、GSE50081 回顾性重分析和单机容量。七联界面图是主 Figure 2，
  不再声明另有 Figure S7。
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
- 新代码/结果版本公开、云端最终同步复测、Word 逐页视觉核对。

最终回复信保持 55 个 Comment、55 个 Response 和 55 个
Changes in the manuscript 段落；本轮新增完成声明由本地代码、图和验证记录支持，
公开仓库对应版本仍须另行发布。
