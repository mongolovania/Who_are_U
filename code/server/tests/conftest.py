"""pytest 配置和共享 fixtures"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.config import Settings, get_settings


@pytest.fixture
def test_settings():
    """测试用配置 — 不依赖真实 API Key"""
    return Settings(
        ai_api_key="test-key",
        ai_base_url="https://api.deepseek.com/anthropic",
        ai_model="deepseek-v4-pro",
        environment="testing",
        debug=True,
    )


@pytest.fixture
def app(test_settings):
    """创建测试用 FastAPI 应用"""
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: test_settings
    return app


@pytest.fixture
async def client(app):
    """异步 HTTP 测试客户端"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
