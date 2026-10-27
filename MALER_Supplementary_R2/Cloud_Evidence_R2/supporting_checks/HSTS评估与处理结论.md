# HSTS 评估与处理结论

检查日期 2026 年 10 月 23 日。结论是本轮保留 HTTPS 与 HTTP 308 重定向，暂不在共享主机名 www.inbirg.com 启用 HSTS。

## 实测依据

- Nginx 配置检查通过。www.inbirg.com 的同一个 80/443 server 承载 MALER 和多项其他应用，MALER 通过独立 include 文件接入。
- 首页和 MALER 四个信息页均通过证书验证的 HTTPS 返回 200；没有 Strict-Transport-Security 响应头。
- 对其他八个应用首页做只读抽样：五个 HTTPS 返回 200，两个 HTTPError、一个超时。部分成功页面仍含同主机 HTTP 绝对链接。
- 抽样没有测试登录、表单、JavaScript、下载和非标准端口，不能据此判断整个共享域名均兼容 HTTPS。失败抽样也不能证明 HSTS 会造成这些失败。

## 为什么没有直接开启

HSTS 是浏览器按主机名记忆的策略，不能仅限制在 /maler/ 路径；不加 includeSubDomains 也仍影响 www.inbirg.com 的其他路径和端口。短 max-age 能缩短缓存时间，但不能隔离项目。

当前用户授权的任务是评估 HSTS。基于实测，不把共享域名策略变更纳入这次 MALER 版本切换。主配置、其他应用和 HSTS 设置均未修改。

## 后续可实施方案

1. 统一核对共享域名下所有应用的 HTTPS 页面、重定向、表单、资源、WebSocket、下载及端口，再在维护窗口逐步增加 max-age；初期不设 includeSubDomains 或 preload。
2. 或为 MALER 配置独立主机名、DNS、证书及代理，验证全部外链和旧地址迁移后，单独启用该主机名的 HSTS。该方案需要额外部署和域名配置，本轮没有执行。

## 投稿表述

HSTS remains disabled on the shared hostname after a read-only review. HTTPS and HTTP-to-HTTPS redirects are verified; compatibility of every co-hosted application has not been established. This is a retained deployment limitation, not a completed security fix.

原始记录见 hsts_assessment.json。标准依据：[RFC 6797 第 5.3 节](https://www.rfc-editor.org/rfc/rfc6797#section-5.3)；主机与端口处理见第 8.3 节。没有进行安全认证或对其他应用实施改动。
