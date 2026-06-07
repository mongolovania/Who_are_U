# 你谁啊 (Who Are U) — 工程代码

## 目录结构

```
code/
├── app/         # Flutter 移动端（iOS + Android）
├── server/      # Python FastAPI 薄后端（AI 代理 + IAP 验证）
├── shared/      # 前后端共享常量和类型
└── h5/          # H5 营销落地页
```

## 技术栈

| 层 | 技术 | 说明 |
|---|------|------|
| 移动端 | Flutter 3.27+ / Dart 3.6+ | 跨平台 UI，主要开发目标 |
| 后端 | Python 3.12+ / FastAPI | 薄代理层，不存储用户数据 |
| 数据库 | SQLite + SQLCipher | 客户端加密存储 |
| AI | Claude API (via DeepSeek) | 对话引擎 |
| 部署 | Docker + Railway/Render | 后端托管 |

## 快速开始

### 前置条件

- Flutter SDK 3.27+
- Python 3.12+
- Docker（可选，用于后端部署）

### 移动端开发

```bash
cd app
flutter pub get
flutter run
```

### 后端开发

```bash
cd server
cp .env.example .env
# 编辑 .env 填入 API Key
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## 相关文档

- [需求设计与商业计划书](E:\98-桌面\02-研究\03-基于用户画像的决策支持app\你谁啊_需求设计与商业计划书-14648f1feb.md)
- [实施任务跟踪](../tasks/todo.md)
- [实施路径规划](C:\Users\mhc\.claude\plans\partitioned-seeking-spark.md)
