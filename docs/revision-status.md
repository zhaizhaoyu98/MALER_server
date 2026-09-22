# MALER 返修状态与发布边界

_基于当前合并分支、46 项测试、公开仓库和 2026-09-22 投稿包重新核对_

## 当前结论

本轮代码侧 P0 已闭环。GitHub 与 Gitee 的 `master` 已合并协作者修改；
`v1.0.0-review2` 用于冻结本次返修版本。代码、验证结果、环境文件、
许可证、表格、主图和补充材料均有可检查产物。

阿里云目前仍运行先前验证的 `13d6b3e`。用户明确要求本轮暂不更新云端，
因此文稿只陈述已经完成的 HTTP 部署验证，不声称线上已运行
`v1.0.0-review2`。HTTPS/TLS 继续作为未完成边界，不列入本轮交付。

## 状态总览

| 领域 | 状态 | 当前证据 |
| --- | --- | --- |
| 统一验证引擎 | 已完成 | 折内插补、缩放、特征处理和嵌套调参 |
| 四类任务验证 | 已完成 | binary、multiclass、regression、survival 结果文件 |
| 外部证据 | 已完成 | GSE37745、GSE50081、GBSG2、TCGA-to-CGGA |
| 特征对齐提示 | 已完成 | 重排列与额外特征页面提示，预测值不变 |
| legacy 公共计算路由 | 已关闭 | `get_cp_combination` 不再注册，返回 404 |
| 自动化测试 | 已完成 | 46/46 通过，Django system check 无问题 |
| 公开仓库 | 已完成 | GitHub 与 Gitee 匿名可读，MIT License |
| 环境说明 | 已完成 | Windows 与 Linux Conda 规格、README 启动步骤 |
| 表格与图片 | 已完成 | Table 1-3、Figure 1-3、Supplementary Figures S1-S7 |
| 投稿 Word | 已完成 | clean/marked 正文与回复信、补充报告、逐页渲染 |
| 阿里云最新版本 | 延后 | 线上为 `13d6b3e`，未更新到 review2 |
| HTTPS/TLS | 明确不做 | 文稿保留 HTTP-only 和非敏感数据边界 |

## 冻结事实

- 默认缺失值处理为训练折内中位数插补，不是均值插补。
- 上传上限为 200 MB；模型上传上限为 100 MB。
- 论文验证设计为外层重复 5 折 10 次、内层 3 折；网页默认重复次数可由环境配置。
- 分类按 balanced accuracy、回归按 R²、生存按 C-index 选择候选。
- 生存排名使用向量化 Cox score statistic，不使用 lifelines `CoxPHFitter` p 值排序。
- PCA 与 no-reduction 均已实现；校准曲线仍未实现。
- 完整 `.maler` 包含拟合后的 Pipeline、特征顺序、任务元数据、manifest 和签名。
- 外部预测按标识符对齐；缺失或重复必需特征会被拒绝，重排和额外特征会显示提示。
- 自动化套件为 46 项。

## 发布物

- GitHub：<https://github.com/zhaizhaoyu98/MALER_server>
- Gitee：<https://gitee.com/zhaoyuzhai/MLSERVER>
- 许可证：MIT
- 发布标签：`v1.0.0-review2`
- Windows 环境：`environment_windows.yml`
- Linux 环境：`environment_server.yml`
- 验证说明：`validation/README.md`
- 部署说明：`deployment/README.md`

审稿意见要求 reproducible scripts or notebooks；本仓库提供脚本、示例数据、
固定随机种子、机器可读结果和运行说明，因此不额外承诺不存在的 notebook。

## 云端验证边界

2026-09-22 已对 `13d6b3e` 完成以下检查：主页、分析、预测、帮助和方法注册表
返回 HTTP 200；Nginx 对任务缓存静态路径返回 404；六项非 TLS 应用控制通过。

以下内容未完成且不得写成已完成：HTTPS、HSTS、安全 Cookie、独立渗透测试、
防火墙审计、静态存储加密、备份恢复演练，以及 review2 的云端部署。

## 投稿前仍需作者确认

1. 在 Word/Zotero 中执行一次 Refresh，不要 Unlink Citations。
2. 核对作者、单位、基金、通讯作者、利益冲突和伦理声明。
3. 按期刊要求确认图片格式、DPI、文件大小和上传类别。
4. 决定可真实履行的服务器维护期限与负责人；不要承诺永久可用。

## 非阻塞后续工作

- 把 24 小时缓存清理命令部署为定时任务，或降低自动删除措辞。
- 增加独立 sklearn/scikit-survival 参考脚本结果对照。
- 清理旧备份模板和孤立邮件模块。
- Python/Django 受支持版本迁移。
- 生存校准曲线、高级插补、正式可用性研究和前瞻性验证。

这些工作不是本次 `v1.0.0-review2` 的完成条件，也不得用虚构数据替代。
