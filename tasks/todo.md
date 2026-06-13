# 你谁啊 (Who Are U) — 实施任务跟踪

> 商业模式：**月付订阅制**（免费下载 + ¥35/月 + ¥299/年 + 3天全功能试用）
> 交付策略：**两阶段 MVP**（v1 本地优先 → v2 账户+同步+智能）
> 架构设计：[架构设计-账户隐私同步智能.md](架构设计-账户隐私同步智能.md)
> WBS 详见：[WBS-产品分解结构.md](WBS-产品分解结构.md)
> 开始日期：2026-06-06

---

## 📋 交付路线图

```
MVP v1 (13w·本地优先)              MVP v2 (+9w·账户+智能)        Phase 2+
──────────────────────────────────────────────────────────────────────→
S1✅ S2🔜  S3     S4     S5      S6     S7     S8    S9     S10+
 2w   2w    4w     3w     3w      3w     2w     2w    2w      3w+
 │    │     │      │      │       │      │      │     │       │
脚手架 画像  对话   山+订阅 测试   账户+  Ombre  聚合  合规    增值
                         +发布   同步   Brain  引擎  +部署   扩展
```

### 为什么分两阶段？
- **MVP v1**：先验证核心价值假设——用户愿意为陪伴+决策付费吗？
- **MVP v2**：再建基础设施——账户绑定/E2E同步/Ombre Brain/聚合/合规
- **技术解耦**：本地加密→E2E同步是无缝升级路径，不破坏现有数据

---

## Phase 1: MVP v1 — 本地优先·核心闭环（目标 13 周）

### Sprint 1: 项目脚手架 — Week 1-2 ✅
- [x] 搭建 `code/` 目录结构（app + server）
- [x] 创建 Flutter 项目骨架（Flutter SDK 3.44.1, Dart 3.12.1）
- [x] 创建 Python 后端骨架（FastAPI, Dockerfile）
- [x] 定义数据模型（MountainNode, User, PersonaTrait, EmotionTag）
- [x] 配置路由（GoRouter，10+ 路由）
- [x] 实现加密存储层（SQLite + sqlcipher, FileStore, SecureStore）
- [x] 配置 CI/CD（GitHub Actions: flutter-ci + server-ci）
- [x] AI API 代理端点（/api/chat, 17 tests ✅）
- [x] 设计 5 阶段对话提示词模板

### Sprint 2: 用户画像 + 初始测试 — Week 3-4 ✅
- [x] 实现加密密钥初始化（SecureStore 生成/获取加密密钥，供数据库和文件加密使用）
- [x] 设计 10 题初始测试（题目内容 + 评分算法 + 3 特质映射）
- [x] 实现测试 UI（渐变背景、题目切换动画）
- [x] 实现特质泡泡展示（PersonaTrait，气泡动画）
- [x] 实现用户画像存储和读取（SQLite CRUD + 加密）
- [x] **Bug 修复**：SQLCipher 加密密钥提前到 onConfigure + PersonaRepository.save 连线
- [x] **Analyzer 清零**：8 errors + 2 warnings 修复，16/16 tests ✅
- [x] **CI/CD**：GitHub Actions + Codemagic 双流水线就绪

### Sprint 3: 陪伴对话引擎 + 决策智囊 — Week 5-8（核心瓶颈·4周）🔜
- [x] ✅ **Memory Palace v9 完整版 已交付**（24模块·L0/L1/L2/L3全层+Track C SOTA增强+v9·~25,000行·27测试·详见偏差分析报告）
- [x] ✅ **Memory Palace v9.0.0 检索引擎修复**（5 Fix·Benchmark驱动·MP #1/8·862 tests green·详见[记忆宫殿详细设计 §7.6](记忆宫殿详细设计.md)）
- [x] ✅ **24原型数据模型**（Archetype + ArchetypeRegistry + 6 trait 映射·含 8 tests）
- [x] ✅ **大师选择引擎**（SelectionEngine + DecisionService·24×50匹配·top 7）
- [ ] 实现 50 大师 JSON 数据库（4领域 × 12-13位 × 方法论+金句+使用指南）
- [ ] ⚠️ **画像演化引擎**（静态快照→动态画像·PersonaTrait.score 随时间/情绪/对话更新·MP信号接入·DDI感知置信度）
- [ ] 实现对话状态机（ConversationEngine·闲聊模式 ↔ 决策模式切换）
- [ ] 实现各阶段处理器（5 个 Stage + 阶段3-4大师注入）
- [ ] 实现对话 UI（ChatBubble, MasterBadge, MasterPanel, TypingIndicator, StageProgress）
- [ ] 实现陪伴角色形象组件（CompanionAvatar·状态变化）
- [ ] 实现 AI API 调用集成（通过后端代理·流式输出）
- [ ] 实现对话历史加密存储（FileStore）
- [ ] 实现超长上下文管理（对话压缩·token预算控制·记忆浮现）
- [ ] 实现决策报告生成和展示（DecisionReportCard·含大师引用标注）
- [ ] 实现离线降级提示
- [ ] **实现会员权限检查（3天免费试用后触发付费墙）**

> ⚠️ **架构边界（2026-06-13）**：Memory Palace v9.0.0 = 记忆引擎·六维跑分均第一·862 tests green。**禁止修改 MP 内部检索/存储/推理逻辑**。画像演化改进全部在 Flutter/Dart 侧——通过 MP API 调用已有信号。

### Sprint 4: 我之山 + 月付订阅 — Week 9-11（3周）

**我之山：**
- [ ] 实现山体 CustomPainter（3 层渐变色：迷雾→暖光→星空）
- [ ] 实现独影剪影绘制（ShadowFigurePainter）
- [ ] 实现节点绘制和定位（NodePainter，基于 position 坐标）
- [ ] 实现可缩放山脉视口（InteractiveViewer + MountainViewport）
- [ ] 实现迷雾效果层（FogLayer，免费层仅少量节点可见）
- [ ] 实现节点详情弹窗（NodeDetailSheet）
- [ ] 实现对话→节点映射逻辑

**月付订阅（仅自动续期订阅）：**
- [ ] 集成 IAP 插件（in_app_purchase，仅自动续期订阅）
- [ ] 实现付费墙 UI（会员权益展示 + 订阅按钮）
- [ ] 实现订阅状态管理（SubscriptionBloc，本地持久化）
- [ ] 实现票据验证后端（Apple/Google 收据验证）
- [ ] 实现免费试用管理（试用到期倒计时）
- [ ] 实现订阅恢复（换机/重装恢复）

### Sprint 5: 基础称号 + 基础外观 + 测试 + 发布 — Week 11-13（3周）

**基础称号（MVP v1 含，月付会员权益）：**
- [ ] 定义 10+ 基础称号（首次决策/十次对话/深夜倾诉者/…）
- [ ] 实现称号触发引擎（事件驱动）
- [ ] 实现称号展示 UI（独影旁称号徽章）

**基础外观（MVP v1 含，月付会员权益）：**
- [ ] 实现 3-5 件基础外观物品（帽子/眼镜/斗篷）
- [ ] 实现独影换装基础引擎（Layer 叠加）
- [ ] 实现外观选择 UI

**测试与发布：**
- [ ] 编写核心业务逻辑单元测试（conversation_engine, mountain, persona）
- [ ] 编写 Widget 测试（chat_bubble, mountain_painter）
- [ ] 编写集成测试（完整对话流程: 画像→对话→报告→节点→付费墙）
- [ ] TestFlight 构建配置
- [ ] 后端部署到 Railway/Render
- [ ] 隐私标签和 App Store 元数据准备（含订阅商品审核）

### 🎯 M4: MVP v1 App Store 提交 — Week 13

---

## Phase 2: MVP v2 — 账户+同步+智能（目标 9 周·Week 14-22）

### Sprint 6: 账户 + E2E 加密同步 — Week 14-16（3周）

**账户系统：**
- [ ] 设计用户表 UserRecord（user_id, email_hash, public_key）
- [ ] 实现注册/登录 API（email_hash + auth_key bcrypt + JWT）
- [ ] 实现 Apple Sign In 集成（sign_in_with_apple + Hide My Email）
- [ ] 实现会话管理（JWT refresh/revoke + 限流 + 防滥用）
- [ ] 实现登录 UI（邮箱输入 + Apple 按钮 + 温暖风格）

**E2E 加密密钥管理：**
- [ ] 实现 Argon2id 密码派生（password → auth_key + encryption_key + recovery_key）
- [ ] 实现 Curve25519 密钥对生成（公钥提交服务端，私钥 SecureStore）
- [ ] 实现 v1→v2 迁移（本地加密密钥 → E2E 密钥材料输入）

**同步协议：**
- [ ] 实现客户端数据打包（SyncObject: id/type/version/encrypted_payload）
- [ ] 实现增量同步（updated_at 版本比较 + 冲突检测）
- [ ] 实现批量同步（首次全量拉取 + 后续增量推送）
- [ ] 实现离线队列（无网络暂存，恢复后自动同步）
- [ ] 实现服务端存储 API（PUT/GET/DELETE 密文 blobs）
- [ ] 实现服务端版本管理 + CRDT merge
- [ ] 实现数据删除（软删除 30 天 → 物理删除）

### Sprint 7: Ombre Brain 深度集成 — Week 17-18（2周）

**多用户改造：**
- [ ] 实现多用户命名空间隔离（user_id → 独立 buckets/ 目录）
- [ ] 实现 bucket 文件 E2E 加密同步（与同步协议集成）
- [ ] 实现跨设备记忆一致性

**对话耦合（breath → hold → dream 循环）：**
- [ ] 实现对话前 breath()：自动浮现未解决记忆 + 高权重记忆 + 上次 feel
- [ ] 实现对话中 hold()：每阶段关键洞察自动标记 + 情感坐标 (valence/arousal)
- [ ] 实现对话后 dream()：自省消化 + feel 沉淀 + 重复议题检测

**决策记忆类型：**
- [ ] 定义 decision bucket 类型（决策结果/选项/选择/后续评价）
- [ ] 实现决策回顾 breath（下次决策前浮现相关历史决策）

**称号联动：**
- [ ] 实现记忆量称号触发（"百段记忆""千思万绪"基于桶数量）
- [ ] 实现情感深度称号触发（"情绪洞察者"基于 valence 变化追踪）

### Sprint 8: 匿名聚合引擎 — Week 19-20（2周）

**差分隐私聚合器：**
- [ ] 实现 DifferentialPrivacyAggregator 核心类
- [ ] 实现计数聚合（按维度统计: 称号/特质/情绪，min_threshold=100）
- [ ] 实现 Laplace 噪声注入（ε=0.5 隐私预算）
- [ ] 实现模糊化展示（分位数表述："超过 90% 的同行者"）
- [ ] 实现隐私预算管理（每用户每维度 ε 追踪 + 耗尽保护）

**称号稀有度 API：**
- [ ] 实现 compute_title_rarity()（计数→加噪→阈值→模糊→输出）
- [ ] 实现稀有度分级（神话 <1% / 史诗 1-5% / 稀有 5-15% / 普通 >15%）
- [ ] 实现缓存策略（每 6 小时刷新）

**隐藏道具发放规则引擎：**
- [ ] 实现绝对稀有度触发（称号持有率 <1% → 解锁隐藏外观）
- [ ] 实现时间窗口触发（节日限定 + 稀有度阈值）
- [ ] 实现匿名比较触发（分位数达标 → 解锁隐藏称号）
- [ ] 实现随机掉落（对话后概率 0.05%-5%，动态调整）
- [ ] 实现情绪共振触发（情绪趋势匹配全局模式 → 解锁限定外观）

### Sprint 9: 全球合规 + 多区域部署 — Week 21-22（2周）

**隐私政策 v2：**
- [ ] 编写中文隐私政策（PIPL 合规 + 数据驻留说明）
- [ ] 编写英文隐私政策（GDPR + CCPA 合规）
- [ ] 编写日文隐私政策（PDPA 合规）

**同意管理系统：**
- [ ] 实现隐私中心 UI（聚合贡献/模型优化/数据导出/删除 开关）
- [ ] 实现同意存储（UserRecord.consents + 审计日志）
- [ ] 实现首次启动渐进式同意流

**数据权利：**
- [ ] 实现数据导出 API（全量 JSON，含解密指引）
- [ ] 实现数据删除 API（软删除 30 天 + 聚合贡献回滚）

**多区域部署：**
- [ ] 部署 eu-west (Frankfurt) — EU/EEA 用户
- [ ] 部署 cn-north (Beijing) — 中国用户（数据不出境）
- [ ] 部署 us-east (Virginia) — 北美用户
- [ ] 部署 ap-southeast (Singapore) — 东南亚/其他用户
- [ ] 实现违规监控 + 通知流程（GDPR 72h / PIPL 立即 / PDPA 72h）

### 🎯 M8: MVP v2 全球上线 — Week 22

---

## Phase 3: 增值扩展（目标 6+ 个月·Week 23+）

### Sprint 10+: 完整称号 + 完整外观 — Week 23-28
- [ ] 隐藏称号设计（20+ 特殊条件触发）
- [ ] 稀有称号限时活动框架
- [ ] 称号收藏墙 UI（全称号展示 + 未获得预览）
- [ ] 聚合引擎联动展示（实时稀有度 + "你在 X% 的同行者中"）
- [ ] 外观物品扩展（20+ 物品定义）
- [ ] 完整外观商店 UI（分类浏览 + 预览试穿 + 已拥有标记）
- [ ] 限定外观（节日限定 + 稀有度解锁 + 隐藏触发）
- [ ] 外观组合系统（多 layer 叠加 + 保存搭配）

### Sprint 11+: 持久记忆 + 独影动画 — Week 29-34
- [ ] 情绪曲线可视化（周/月度情绪变化，Ombre Brain 数据源）
- [ ] 时光轴视图（TimelineScreen，节点时间排列）
- [ ] 年度回顾报告（数据汇总 + 可视化 + 温暖文案）
- [ ] 分享功能（模糊版截图、社交分享卡片）
- [ ] 动画引擎选型与集成（Spine vs Rive）
- [ ] 7 种困难动作动画资源制作
- [ ] AI 困难检测逻辑 + 纪念碑定格

### Sprint 12+: 副峰 + Android + 运营后台 — Week 35-44
- [ ] 副峰创建和管理（SubPeak）
- [ ] 副峰 UI + 主副峰联动
- [ ] Android 适配（Material Design 微调）
- [ ] Google Play 上线
- [ ] 运营后台（称号/外观配置、活动管理、聚合数据看板）

---

## 里程碑总览

| 里程碑 | 时间 | 交付物 | 状态 |
|--------|------|--------|------|
| **M0** | Week 2 | 脚手架完成 | ✅ |
| **M1** | Week 6 | Memory Palace v9 底座就绪 → 对话引擎 → 报告 | 🔜 |
| **M2** | Week 8 | 免费试用闭环 — 2次免费对话后触发付费墙 | ⬜ |
| **M3** | Week 11 | 订阅商业闭环 — 我之山 + 月付订阅 + 基础称号外观 | ⬜ |
| **M4** | Week 13 | 🚀 MVP v1 App Store 提交审核 | ⬜ |
| **M5** | Week 16 | 账户+同步闭环 — 跨设备数据自动同步 | ⬜ |
| **M6** | Week 18 | MP 生产连线+对话耦合 — breath/hold/dream完整集成 | ⬜ |
| **M7** | Week 20 | 聚合引擎上线 — 称号稀有度% + 隐藏道具 | ⬜ |
| **M8** | Week 22 | 🚀 MVP v2 全球上线（4 区域部署） | ⬜ |
| **M9** | Week 44+ | 🚀 完整产品上线（含 Android） | ⬜ |

---

## 当前阻塞项

_（无）_

## 技术债务

- [ ] **app.py 生产连线**：`get_orchestrator()` 未注入 v9 增强模块（DDA+graph+L2+Track C），API 使用 basic fallback
- [ ] **Prompt 缓存**：静态系统 Prompt 缓存未实现（设计目标节省 30-50% 输入 token）
- [ ] **流式 SSE 端点**：LLMGateway.chat_stream() 存在，api_router 未暴露
- [ ] **对话压缩器**：ConversationCompressor 类未创建
- [ ] 数据库加密密钥实际集成（目前为 TODO）
- [ ] IAP 票据验证后端实现（目前为占位）

## 待决策项

- [x] Flutter vs React Native 最终确认 — 已选择 Flutter
- [x] 商业模式 — 免费下载 + ¥35/月 + ¥299/年 + 3天全功能试用
- [x] 交付策略 — 两阶段 MVP（v1 本地优先 → v2 账户+同步+智能）
- [x] 认证方式 — Email + Apple Sign In
- [x] 加密方案 — E2E AES-256-GCM + Argon2id + Curve25519
- [x] 聚合方案 — 差分隐私（Laplace noise, ε=0.5）
- [x] 数据驻留 — 4 区域部署（eu/cn/us/sg）
- [x] 用户画像 — 24原型系统（4阵营×6）+ 6 trait兼容映射
- [x] 决策智囊 — 50大师蒸馏引擎 + 选择引擎
- [x] 记忆引擎 — Memory Palace v9 完整版·v1 已交付（17专家6轮辩论定版·24模块·Track C增强）
- [x] 交互架构 — 我之山3D主场景 + 对话页面 + 装扮页面
- [ ] Spine vs Rive 动画引擎选择（Phase 3 决策）
- [ ] 3D 渲染方案选择（Flutter 3D/Unity嵌入/Rive/CustomPainter）（Sprint 4 前决策）
- [ ] 后端部署平台选择（Railway / Render / 自托管）（Sprint 5 前决策）

---

## Sprint 2 优先任务 (Week 3-4) 🔜

| # | 任务 | 估时 | 依赖 |
|---|------|------|------|
| 1 | 加密密钥初始化（SecureStore） | 0.5d | — |
| 2 | 10题测试内容设计 + 评分算法 | 2d | — |
| 3 | 测试 UI（渐变+动画） | 2d | #2 |
| 4 | 特质泡泡展示 | 2d | #3 |
| 5 | 画像存储 CRUD | 1d | #1 |

---

_最后更新：2026-06-10（Memory Palace v9 完整版 ✅·v9.0.0 检索引擎修复 ✅·5 Fix·MP #1/8·862 tests green·偏差分析+技术方案同步完成）_
