## Project Structure: Twin-Repo Architecture

本项目的**文档/规划**与**工程实现**分离为两个关联文件夹：

| 用途 | 路径 | 存放内容 |
|------|------|----------|
| 📋 文档 & 规划 | `E:\98-桌面\02-研究\03-基于用户画像的决策支持app\` | BP、商业计划书、产品策划、需求设计、市场研究、用户研究 |
| 💻 工程 & 代码 | `E:\01-Project\01-Who_are_U\`（本目录） | 所有工程代码、技术实现、测试、部署配置 |

### 关联规则
- 编码前先查阅文档库中的 BP 和需求设计，确保实现与产品规划一致
- 文档库中的技术约束和架构决策需同步到本仓库的 `tasks/` 或设计文档中
- 工程实现中的技术发现（可行性、成本）需反馈到文档库的产品决策中
- 两个文件夹的 `CLAUDE.md` 互相引用，保持上下文同步

### 核心参考文档
- [你谁啊_需求设计与商业计划书](E:\98-桌面\02-研究\03-基于用户画像的决策支持app\你谁啊_需求设计与商业计划书-14648f1feb.md)
- [记忆系统](E:\98-桌面\02-研究\03-基于用户画像的决策支持app\记忆系统.docx)
- [记忆系统-实现vs方案-偏差分析](tasks/记忆系统-实现vs方案-偏差分析.md) — **2026-06-10 更新**：Memory Palace v9 完整版已交付（24模块·~25,000行），远超原 v1 降级方案
- [记忆宫殿详细设计](tasks/记忆宫殿详细设计.md) — 已同步更新至实际实现状态

---

## Workflow Orchestration



### 1. Plan Node Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)

- If something goes sideways, STOP and re-plan immediately - don't keep pushing

- Use plan mode for verification steps, not just building

- Write detailed specs upfront to reduce ambiguity



### 2. Subagent Strategy

- Use subagents liberally to keep main context window clean

- Offload research, exploration, and parallel analysis to subagents

- For complex problems, throw more compute at it via subagents

- One tack per subagent for focused execution



### 3. Self-Improvement Loop

- After ANY correction from the user: update `tasks/lessons.md` with the pattern

- Write rules for yourself that prevent the same mistake

- Ruthlessly iterate on these lessons until mistake rate drops

- Review lessons at session start for relevant project



### 4. Verification Before Done

- Never mark a task complete without proving it works

- Diff behavior between main and your changes when relevant

- Ask yourself: "Would a staff engineer approve this?"

- Run tests, check logs, demonstrate correctness



### 5. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"

- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"

- Skip this for simple, obvious fixes - don't over-engineer

- Challenge your own work before presenting it



### 6. Autonomous Bug Fixing

- When given a bug report: just fix it. Don't ask for hand-holding

- Point at logs, errors, failing tests - then resolve them

- Zero context switching required from the user

- Go fix failing CI tests without being told how



## Task Management



1. **Plan First**: Write plan to `tasks/todo.md` with checkable items

2. **Verify Plan**: Check in before starting implementation

3. **Track Progress**: Mark items complete as you go

4. **Explain Changes**: High-level summary at each step

5. **Document Results**: Add review section to `tasks/todo.md`

6. **Capture Lessons**: Update `tasks/lessons.md` after corrections



## Core Principles



- **Simplicity First**: Make every change as simple as possible. Impact minimal code.

- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.

- **Minimat Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

