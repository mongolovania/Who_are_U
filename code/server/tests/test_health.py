"""健康检查端点测试"""

import pytest


@pytest.mark.anyio
async def test_health_endpoint(client):
    """测试 /health 返回 200"""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "who-are-u-api"


@pytest.mark.anyio
async def test_ready_endpoint(client, test_settings):
    """测试 /health/ready 在 API Key 配置时返回 ready"""
    response = await client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
