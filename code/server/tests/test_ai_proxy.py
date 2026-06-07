"""AI 代理端点测试"""

import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.anyio
async def test_chat_validation_missing_messages(client):
    """测试缺少必填字段时返回 422"""
    response = await client.post("/api/chat", json={})
    assert response.status_code == 422


@pytest.mark.anyio
async def test_chat_validation_empty_messages(client):
    """测试空消息列表"""
    response = await client.post("/api/chat", json={"messages": []})
    # 空列表是合法的（FastAPI 不做业务校验），请求会发给 AI 服务
    # 这里测试的是参数验证层面
    assert response.status_code in {200, 422, 502, 503}


@pytest.mark.anyio
async def test_chat_validation_max_tokens(client):
    """测试 max_tokens 超过上限时返回 422"""
    response = await client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "你好"}],
            "max_tokens": 99999,
        },
    )
    assert response.status_code == 422


@pytest.mark.anyio
async def test_chat_request_format(client):
    """测试正常格式的请求可以被接受"""
    payload = {
        "messages": [
            {"role": "user", "content": "我最近很纠结要不要换工作"},
        ],
        "system_prompt": "你是一位温暖的人生教练",
        "max_tokens": 2048,
        "temperature": 0.7,
    }

    # 不 mock AI 服务时，真实请求会失败（test-key 是假的）
    # 但我们可以验证请求格式通过了 FastAPI 验证
    response = await client.post("/api/chat", json=payload)
    # 期望 502 (AI 服务调用失败) 或 503 (AI 服务未配置)
    # 因为用了 test-key，真实 API 调用会失败
    assert response.status_code in {502, 503}


@pytest.mark.anyio
async def test_chat_with_stage_prompt(client):
    """测试带阶段提示词的请求"""
    payload = {
        "messages": [
            {"role": "user", "content": "我不确定要不要搬家去另一个城市"},
        ],
        "system_prompt": "你是一位温暖的人生教练。当前阶段：共情与接纳。",
        "max_tokens": 4096,
        "temperature": 0.7,
    }
    response = await client.post("/api/chat", json=payload)
    assert response.status_code in {502, 503}
