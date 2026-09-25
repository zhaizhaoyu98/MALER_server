# MALER 返修状态与发布边界

_2026-09-24–25 一致性复核、阿里云部署后复测与 Word/Zotero 核验；以源码、结果、投稿包和部署记录为准_

## 当前结论

本轮已完成四任务患者级分区审计与分析重算，并同步更新网页解析、主图、
补充图、表格、正文和回复信。新本地投稿包位于
`article-review/revision/final_submission_patient_level_20260923/`。
当前源码、验证结果和图件已公开为 `v1.0.0-review4`；`v1.0.0-review3`
是此前基线，**不包含**本轮修正。投稿 DOCX 仍位于本地包，不随代码仓库公开。
2026-09-24 对 clean/marked 文件接受修订后的逐段文字、11 个审稿 DOI 的真实正文
引用、81 条 Zotero 文献、四任务与两份外部队列原文件哈希进行了复核；记录见投稿包
`99_internal_qa/alignment_audit_20260924.md`。旧审计报告所指 `temp` 四份 DOCX
当前未找到，故仅对现有投稿包作出一致性结论。

同日对新增的 `reviewer-editor-report.md` 做了实质核查：历史拆分来源无法追溯、
GSE50081 并非新未触碰队列、特征选择并未普遍提高预测性能这三项是真实限制。
本地手稿已把嵌套 CV 放到摘要首位，将两项 GEO 结果明确为回顾性复算，并把
特征选择定位为候选压缩与优先排序。独立两模型 scikit-learn 对照和 Table 1–3/S1
原本已存在，不应误报为缺失。逐项结论见投稿包
`99_internal_qa/reviewer_editor_report_audit_20260924.md`。

阿里云旧版 `13d6b3e` 已于 2026-09-24 快进到患者级修正版应用源码
`35aa116`，并重新加载 Gunicorn 工作进程。独立服务器工作树通过 Django 检查、
50 项网站测试和 3 项患者分区测试；公网主页、分析、预测、帮助页返回 HTTP 200，
分析和预测页均可见授权确认控件。这是页面级冒烟检查，不等于线上端到端上传
或完整安全审计。HTTPS/TLS 仍未验证，线上仅 HTTP。

## 状态总览

| 领域 | 状态 | 当前证据 |
| --- | --- | --- |
| 统一验证引擎 | 已完成 | 折内插补、缩放、特征处理和嵌套调参 |
| 四类任务验证 | 本地已重算 | 患者级 binary 309/299、multiclass 482/468、regression 367/162、survival 160/133 |
| 外部证据 | 本地已重算 | GSE37745、GSE50081 回顾性重分析、GBSG2、TCGA-to-CGGA |
| 特征对齐提示 | 已完成 | 重排列与额外特征页面提示，预测值不变 |
| legacy 公共计算路由 | 已关闭 | `get_cp_combination` 不再注册，返回 404 |
| 自动化测试 | 已完成本地测试；云端待本轮复测 | 旧云端版本曾通过 50/50 Django 与 3/3 患者分区测试；本轮本地 54/54 Django 与 3/3 患者分区测试通过，Django system check 无问题 |
| 独立参考实现对照 | 已完成两项 | 直接 scikit-learn 重拟合二分类 Logistic 与回归 Ridge；特征、留出预测及指标与归档结果一致；不外推到全部估计器 |
| 上传授权及帮助页 | 已部署并完成页面级复测 | 两类上传均需确认，服务端自动测试拒绝无确认提交；公网两页可见控件；Help 修正文案 |
| 本地浏览器验收 | 既有界面已完成；本轮新控件待鼠标复看 | 2026-09-23 的 27/27 鼠标交互通过；本轮新增授权框和帮助页改动已通过 Django 页面/拒绝测试，但尚未重新鼠标验收 |
| 签名模型闭环 | 已完成 | 本地生成、下载、重新上传 `.maler` 并完成 50 个样本预测 |
| 公开仓库 | 已完成 | GitHub 与 Gitee 匿名可读，MIT License |
| 环境说明 | 已完成 | Windows 与 Linux Conda 规格、README 启动步骤 |
| 表格与图片 | 已更新 | Table 1-3、Table S1、Figure 1-3、Supplementary Figures S1-S6；Figure 2 已重抓缩放/预测隐私面板并移除旧“24 小时物理删除”截图，不再引用不存在的 S7 |
| 投稿 Word | 本轮已重新核验 | 2026-09-25 对四份核心 DOCX 重做 LibreOffice 全页渲染，并用 Word 导出当前 clean 手稿与回复信预览核对关键页；Zotero Refresh 在当前手稿逐字节副本上通过，51 个字段、81 条书目和文件 SHA-256 保留。作者仍须核对投稿元数据 |
| 新版本公开 | 本轮待完成 | 患者级修正已在 GitHub/Gitee 的 `v1.0.0-review4`，阿里云此前到 `35aa116`；2026-09-25 新增的上传大小、重复表头和分析页隐私修复尚未提交发布或上云 |
| HTTPS/TLS | 未验证 | 文稿保留 HTTP-only 和非敏感数据边界 |

## 冻结事实

- 默认缺失值处理为训练折内中位数插补，不是均值插补。
- 上传上限为 200 MB；模型上传上限为 100 MB。
- 论文脚本外层重复 5 折 10 次、内层 3 折；网页默认外层重复 5 折 2 次。
- 分类按 balanced accuracy、回归按 R²、生存按 C-index 选择候选。
- 生存排名使用向量化 Cox score statistic，不使用 lifelines `CoxPHFitter` p 值排序。
- PCA 与 no-reduction 均已实现；校准曲线仍未实现。
- 完整 `.maler` 包含拟合后的 Pipeline、特征顺序、任务元数据、manifest 和签名。
- 外部预测按标识符对齐；缺失或重复必需特征会被拒绝，重排和额外特征会显示提示。
- 旧云端部署的自动化套件为 50 项 Django 测试，另有 3 项患者分区测试；本轮本地修复后 Django 套件增至 54 项，仍有 3 项患者分区测试。
- 独立参考对照见 `validation/reference_parity_check.py` 和 `validation/results/reference_parity_check.json`；仅覆盖两种已锁定模型。
- 2026-09-23 的 1440px Edge 验收覆盖 Analysis、Predict、Help、预览、结果页和项目删除；临时本地签名密钥未写入仓库。

## 发布物

- GitHub：<https://github.com/zhaizhaoyu98/MALER_server>
- Gitee：<https://gitee.com/zhaoyuzhai/MLSERVER>
- 许可证：MIT
- 旧发布标签：`v1.0.0-review3`（不含患者级修正）
- 当前发布标签：`v1.0.0-review4`（患者级修正、机器可读结果和图件）
- Windows 环境：`environment_windows.yml`
- Linux 环境：`environment_server.yml`
- 验证说明：`validation/README.md`
- 部署说明：`deployment/README.md`

审稿意见要求 reproducible scripts or notebooks；本仓库提供脚本、示例数据、
固定随机种子、机器可读结果和运行说明，因此不额外承诺不存在的 notebook。

## 云端验证边界

2026-09-22 已对 `13d6b3e` 完成以下检查：主页、分析、预测、帮助和方法注册表
返回 HTTP 200；Nginx 对任务缓存静态路径返回 404；六项非 TLS 应用控制通过。

2026-09-24，先在独立服务器工作树对 `35aa116` 完成 Django system check、
50/50 Django 测试与 3/3 患者分区测试，然后在生产项目中仅做 fast-forward，
保留旧版回退分支，重新加载 Gunicorn workers。公网四个页面返回 HTTP 200；
分析与预测页均可见 `data_consent` 控件。未执行公网真实上传或再次验证
生产清理计划、备份和安全边界；以上页面检查不能替代这些项目。

以下内容未完成且不得写成已完成：HTTPS、HSTS、安全 Cookie、独立渗透测试、
防火墙审计、静态存储加密、备份恢复演练，以及线上端到端上传/删除演练。

## 投稿前仍需完成或由作者确认

1. 当前 clean 手稿已在逐字节副本上重新用 Word/Zotero Refresh，50 个引文、1 个书目字段、81 条书目保留；刷新副本与正式稿 SHA-256 同为 `5BA03C37453AE74A3003660BFAA4A785F01F2602E54192249C521F3851564557`。Word 和 LibreOffice 均已渲染本轮稿件，勿 Unlink Citations。
2. 核对作者、单位、基金、通讯作者、利益冲突和伦理声明，以及期刊图片格式、DPI、文件大小、上传类别。
3. 先提交发布并部署本轮本地三项修复，再用非敏感合成数据复核分析上传超限、重复样本表头和两页隐私提示。若期刊要求实际线上任务证据，再做端到端上传/删除演练；生产定时清理、备份策略和维护负责人仍须运营方确认，不承诺永久可用。

## 非阻塞后续工作

- 把 24 小时缓存清理命令部署为定时任务，或降低自动删除措辞。
- 增加独立 sklearn/scikit-survival 参考脚本结果对照。
- 清理旧备份模板和孤立邮件模块。
- Python/Django 受支持版本迁移。
- 生存校准曲线、高级插补、正式可用性研究和前瞻性验证。

这些工作不是本次本地患者级修订的完成条件，也不得用虚构数据替代。
