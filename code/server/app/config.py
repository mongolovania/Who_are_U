"""应用配置管理

使用 pydantic-settings 从环境变量加载配置。
所有敏感值必须通过环境变量提供，无默认值。
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 运行环境
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # AI 服务
    ai_api_key: str = ""
    ai_base_url: str = "https://api.deepseek.com/anthropic"
    ai_model: str = "deepseek-v4-pro"
    ai_max_tokens: int = 4096
    ai_temperature: float = 0.7

    # IAP 验证
    app_store_shared_secret: str = ""
    app_store_environment: str = "sandbox"  # sandbox | production

    # 速率限制
    rate_limit_per_minute: int = 20
    rate_limit_per_hour: int = 200

    # 服务器
    host: str = "0.0.0.0"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
