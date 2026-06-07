"""支付路由 — IAP 收据验证

职责：
- 验证 App Store 票据（production/sandbox）
- 验证 Google Play 票据
- 返回验证结果
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class ReceiptVerifyRequest(BaseModel):
    """票据验证请求"""

    platform: str = Field(..., pattern="^(ios|android)$")
    receipt: str = Field(..., description="Base64 编码的收据")
    product_id: str


class ReceiptVerifyResponse(BaseModel):
    """票据验证响应"""

    valid: bool
    product_id: str | None = None
    purchase_date: str | None = None
    expires_date: str | None = None
    transaction_id: str | None = None
    error: str | None = None


@router.post("/verify", response_model=ReceiptVerifyResponse)
async def verify_receipt(request: ReceiptVerifyRequest):
    """验证 IAP 收据

    POST /api/payment/verify

    验证从客户端提交的应用内购买票据。
    """
    if request.platform == "ios":
        return await _verify_apple_receipt(request)
    elif request.platform == "android":
        return await _verify_google_receipt(request)

    raise HTTPException(status_code=400, detail="不支持的平台")


async def _verify_apple_receipt(
    request: ReceiptVerifyRequest,
) -> ReceiptVerifyResponse:
    """验证 Apple App Store 票据"""
    # TODO: 实现 Apple App Store 票据验证
    # 使用 app_store_shared_secret 调用 Apple verifyReceipt 端点
    return ReceiptVerifyResponse(
        valid=False,
        error="Apple IAP 验证暂未实现",
    )


async def _verify_google_receipt(
    request: ReceiptVerifyRequest,
) -> ReceiptVerifyResponse:
    """验证 Google Play 票据"""
    # TODO: 实现 Google Play 票据验证
    return ReceiptVerifyResponse(
        valid=False,
        error="Google Play 验证暂未实现",
    )
