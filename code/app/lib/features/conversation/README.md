# 决策助手对话模块 (Conversation)

核心功能模块。实现 AI 引导式的 5 阶段深度对话。

## 架构

```
presentation/  ←── UI 层（Bloc + Widget）
domain/        ←── 业务逻辑层（状态机 + 阶段处理器）
data/          ←── 数据层（模型 + 仓库 + 提示词模板）
```

## 5 阶段对话流程

```
EMPATHY(共情) → SELF_SCAN(自我扫描) → OPTION_BREAKDOWN(选项拆解) → INNER_CONFIRM(内在确认) → DECISION_OUTPUT(决策输出)
```

每阶段有独立的提示词模板，AI 根据用户输入判断是否进入下一阶段。

## 关键文件

- [conversation_engine.dart](domain/conversation_engine.dart) — 对话状态机
- [stage_handler.dart](domain/stage_handler.dart) — 阶段处理接口
- [stages/](domain/stages/) — 5 个阶段具体实现
- [prompt_templates.dart](data/prompt_templates.dart) — 提示词模板
- [conversation_cubit.dart](presentation/cubit/conversation_cubit.dart) — UI 状态管理
