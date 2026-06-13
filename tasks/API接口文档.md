# 你谁啊 (Who Are U) — API 接口文档 v1.0

> 日期：2026-06-08
> 依赖：[总体技术方案](总体技术方案.md) · [软件详细方案](软件详细方案.md)
> 定位：前后端接口契约，可直接生成 OpenAPI 3.0 spec

---

## 一、基础约定

### 1.1 环境

| 环境 | Base URL |
|------|---------|
| 开发 | `http://localhost:8000` |
| 生产 (eu-west) | `https://api-eu.whoareu.app` |
| 生产 (cn-north) | `https://api-cn.whoareu.app` |
| 生产 (us-east) | `https://api-us.whoareu.app` |
| 生产 (ap-southeast) | `https://api-sg.whoareu.app` |

### 1.2 通用请求头

```
Content-Type: application/json
Accept: application/json
Accept-Language: zh-CN | en | ja
X-Client-Version: 1.0.0
X-Platform: ios | android
```

### 1.3 认证

MVP v1（本地方案）：无需认证。请求以设备匿名 ID 标识。
MVP v2（账户方案）：`Authorization: Bearer <JWT>`

### 1.4 通用响应格式

```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "meta": {
    "request_id": "uuid",
    "server_time": "2026-06-08T12:00:00Z"
  }
}
```

错误响应：

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "RATE_LIMITED",
    "message": "请求过于频繁，请稍后再试",
    "details": { "retry_after": 30 }
  },
  "meta": { "request_id": "uuid", "server_time": "..." }
}
```

---

## 二、健康检查

### GET /health

系统健康检查。

**响应 200：**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "uptime_seconds": 86400,
    "dependencies": {
      "llm_api": "healthy",
      "database": "healthy"
    }
  }
}
```

---

## 三、AI 对话接口（核心）

### POST /api/chat

发送消息并获取 AI 回复（流式）。

**请求：**
```json
{
  "device_id": "uuid-v4",                    // 设备匿名ID (v1)
  "conversation_id": "uuid-v4 | null",       // 新对话为null
  "message": "我最近在考虑要不要换工作...",
  "mode": "chat",                            // "chat" | "decision"
  "user_profile": {
    "archetypes": ["strategist", "explorer"],  // 用户原型（最多3个）
    "traits": [                               // Sprint 2兼容
      { "key": "adventurer", "score": 0.8 }
    ]
  },
  "context": {
    "decision_domain": "career",             // 决策领域（可选·NLP自动检测）
    "active_master_ids": [                    // 当前参谋（可选·不传则由服务端选择）
      "charlie_munger", "steve_jobs", "ray_dalio",
      "deng_xiaoping", "nietzsche", "wang_yangming", "musk"
    ],
    "breath_memories": "用户之前提到对工作失去热情...",  // Memory Palace浮现
    "user_mentioned_masters": ["乔布斯"]       // 用户显式提到的大师
  },
  "options": {
    "stream": true,                           // 流式输出
    "model": "deepseek-v3",                   // 模型选择
    "max_tokens": 4096
  }
}
```

**响应 200（流式·SSE）：**
```
data: {"type":"stage","stage":"empathy"}

data: {"type":"token","content":"我"}

data: {"type":"token","content":"听"}

data: {"type":"token","content":"到"}

data: {"type":"token","content":"了"}

...

data: {"type":"master_badge","master":{"id":"charlie_munger","name":"查理·芒格","methodology":"逆向思维"}}

...

data: {"type":"done","usage":{"input_tokens":15000,"output_tokens":2000}}
```

SSE 事件类型：

| type | 说明 | 数据 |
|------|------|------|
| `stage` | 阶段切换（仅决策模式） | `{ "stage": "empathy" \| "self_scan" \| ... }` |
| `token` | 逐 token 输出 | `{ "content": "..." }` |
| `master_badge` | 大师引用标记 | `{ "master": { "id", "name", "methodology" } }` |
| `done` | 对话完成 | `{ "usage": { "input_tokens", "output_tokens" } }` |
| `error` | 错误 | `{ "code": "...", "message": "..." }` |

### GET /api/conversations

获取对话列表。

**查询参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| device_id | string | 设备ID |
| limit | int | 每页条数（默认20） |
| offset | int | 偏移 |

**响应 200：**
```json
{
  "success": true,
  "data": {
    "conversations": [
      {
        "id": "uuid",
        "title": "关于换工作的纠结",
        "mode": "decision",
        "active_master_ids": ["charlie_munger", "steve_jobs", ...],
        "message_count": 24,
        "created_at": "2026-06-08T10:00:00Z",
        "updated_at": "2026-06-08T11:30:00Z"
      }
    ],
    "total": 42
  }
}
```

### GET /api/conversations/{conversation_id}

获取单个对话详情（含完整消息列表）。

**响应 200：**
```json
{
  "success": true,
  "data": {
    "id": "uuid",
    "messages": [
      {
        "id": "msg-001",
        "role": "user",
        "content": "我最近在考虑要不要换工作...",
        "timestamp": "2026-06-08T10:00:00Z"
      },
      {
        "id": "msg-002",
        "role": "companion",
        "content": "听起来你在面对一个挺重要的选择...",
        "master_badge": null,
        "timestamp": "2026-06-08T10:00:05Z"
      }
    ],
    "mode": "decision",
    "archetypes": ["strategist", "explorer"],
    "active_master_ids": ["charlie_munger", ...],
    "created_at": "...",
    "updated_at": "..."
  }
}
```

### DELETE /api/conversations/{conversation_id}

删除对话。

**响应 200：**
```json
{
  "success": true,
  "data": { "deleted": true }
}
```

---

## 四、大师系统（Sprint 3 新增）

### GET /api/masters

获取大师库。

**查询参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| domain | string? | 按领域过滤：economics / philosophy / relationships / career |
| search | string? | 按名字/方法论搜索 |
| limit | int | 默认50 |

**响应 200：**
```json
{
  "success": true,
  "data": {
    "masters": [
      {
        "id": "charlie_munger",
        "name_cn": "查理·芒格",
        "name_en": "Charlie Munger",
        "primary_domain": "economics",
        "methodology": ["逆向思维", "多元思维模型", "能力圈"],
        "one_liner": "反过来想，总是反过来想",
        "description": "伯克希尔·哈撒韦副主席，巴菲特的长期合伙人...",
        "compatible_archetypes": ["strategist", "hermit", "sentinel"]
      }
    ],
    "total": 50,
    "domains": ["economics", "philosophy", "relationships", "career"]
  }
}
```

### POST /api/masters/select

大师选择引擎。根据用户原型+决策领域，返回推荐的大师列表。

**请求：**
```json
{
  "archetypes": ["strategist", "explorer"],
  "domain": "career",
  "user_picks": ["steve_jobs"],
  "exclude_ids": ["charlie_munger"],
  "count": 7
}
```

**响应 200：**
```json
{
  "success": true,
  "data": {
    "selected_masters": [
      {
        "id": "steve_jobs",
        "name_cn": "史蒂夫·乔布斯",
        "match_reason": "user_pick",
        "score": 1.0
      },
      {
        "id": "musk",
        "name_cn": "埃隆·马斯克",
        "match_reason": "archetype_match",
        "score": 0.92
      }
    ],
    "domain": "career",
    "total_available": 12
  }
}
```

---

## 五、决策报告

### POST /api/reports

生成决策报告。汇总本次对话，结构化输出。

**请求：**
```json
{
  "conversation_id": "uuid",
  "options": {
    "include_master_citations": true,
    "format": "structured"
  }
}
```

**响应 200：**
```json
{
  "success": true,
  "data": {
    "report": {
      "id": "uuid",
      "conversation_id": "uuid",
      "summary": "你在考虑是否从大公司跳槽到创业公司...",
      "options": [
        {
          "label": "留在现公司",
          "pros": ["稳定收入", "熟悉环境"],
          "cons": ["缺乏成长", "热情消退"],
          "master_perspectives": [
            { "master": "查理·芒格", "view": "逆向思考——如果留下，5年后你会后悔吗？" }
          ]
        }
      ],
      "recommendation": "基于你的价值观（成长>稳定）和当前状态...",
      "master_citations": [
        { "master": "查理·芒格", "methodology": "逆向思维", "applied_to": "选项A风险分析" }
      ]
    }
  }
}
```

---

## 六、支付验证

### POST /payment/verify-receipt

验证 App Store / Google Play 收据。

**请求：**
```json
{
  "device_id": "uuid",
  "platform": "ios",
  "receipt": "base64-encoded-receipt-data",
  "product_id": "com.whoareu.subscription.monthly"
}
```

**响应 200：**
```json
{
  "success": true,
  "data": {
    "valid": true,
    "subscription": {
      "status": "active",
      "plan": "monthly",
      "expires_at": "2026-07-08T12:00:00Z",
      "auto_renew": true,
      "trial_used": false
    }
  }
}
```

### GET /payment/subscription-status

查询订阅状态。

**查询参数：** `device_id`

**响应 200：**
```json
{
  "success": true,
  "data": {
    "status": "active",
    "plan": "yearly",
    "expires_at": "2027-06-08T12:00:00Z",
    "days_remaining": 365,
    "trial_days_left": 0
  }
}
```

---

## 七、同步接口（MVP v2）

### PUT /sync/objects

上传加密同步对象。

**请求（需认证）：**
```json
{
  "objects": [
    {
      "id": "uuid",
      "type": "conversation_history",
      "version": 3,
      "updated_at": "2026-06-08T12:00:00Z",
      "encrypted_payload": {
        "ciphertext": "base64...",
        "nonce": "base64...",
        "algorithm": "AES-256-GCM"
      },
      "metadata": {
        "size_bytes": 12345,
        "content_hash": "sha256..."
      }
    }
  ]
}
```

### GET /sync/objects

拉取同步对象（增量）。

**查询参数：** `since` (ISO datetime), `type` (可选), `limit` (默认100)

### DELETE /sync/objects

删除同步对象（软删除30天 → 物理删除）。

---

## 八、聚合接口（MVP v2）

### GET /aggregation/title-rarity/{title_id}

获取称号稀有度（差分隐私处理）。

**响应 200：**
```json
{
  "success": true,
  "data": {
    "title_id": "midnight_confidant",
    "rarity_percent": 3.2,
    "rarity_tier": "史诗",
    "total_users_pool": "10000+",
    "privacy_epsilon": 0.5,
    "last_updated": "2026-06-08T06:00:00Z"
  }
}
```

---

## 九、数据权利接口（MVP v2·需认证）

### POST /data/export

请求数据导出。

### POST /data/delete

请求账户删除（30天软删除）。

---

## 十、通用错误码

| HTTP | code | 说明 |
|------|------|------|
| 400 | `INVALID_REQUEST` | 请求格式错误 |
| 400 | `VALIDATION_ERROR` | 参数校验失败 |
| 401 | `UNAUTHORIZED` | 未认证（v2） |
| 403 | `FORBIDDEN` | 权限不足 |
| 404 | `NOT_FOUND` | 资源不存在 |
| 429 | `RATE_LIMITED` | 请求频率超限 |
| 500 | `INTERNAL_ERROR` | 服务端错误 |
| 503 | `LLM_UNAVAILABLE` | LLM 服务不可用 |
| 503 | `SERVICE_UNAVAILABLE` | 服务降级 |

## 十一、限流策略

| 接口 | 限制 | 窗口 |
|------|------|------|
| POST /api/chat | 30 次 | 每分钟 |
| POST /api/chat | 200 次 | 每天 |
| GET /api/conversations | 60 次 | 每分钟 |
| POST /api/masters/select | 30 次 | 每分钟 |
| POST /api/reports | 10 次 | 每小时 |
| POST /payment/verify-receipt | 10 次 | 每分钟 |
| GET /aggregation/* | 30 次 | 每分钟 |

---

*本文档与 `code/server/` 实现同步维护。接口变更需更新本文档版本号。*
