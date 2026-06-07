/// 应用错误类型定义
///
/// 所有业务错误继承自 [AppFailure]，
/// 统一管理错误码和用户友好消息。

import 'package:equatable/equatable.dart';

/// 应用失败基类
sealed class AppFailure extends Equatable {
  final String code;
  final String message;
  final String? userMessage;
  final dynamic originalError;

  const AppFailure({
    required this.code,
    required this.message,
    this.userMessage,
    this.originalError,
  });

  @override
  List<Object?> get props => [code, message, userMessage];
}

// ─── 存储失败 ────────────────────────────────────

final class DatabaseFailure extends AppFailure {
  const DatabaseFailure({
    required super.message,
    super.userMessage = '数据存储出现问题，请稍后重试',
    super.originalError,
  }) : super(code: 'DB_ERROR');
}

final class FileStorageFailure extends AppFailure {
  const FileStorageFailure({
    required super.message,
    super.userMessage = '文件读写失败',
    super.originalError,
  }) : super(code: 'FILE_ERROR');
}

final class EncryptionFailure extends AppFailure {
  const EncryptionFailure({
    required super.message,
    super.userMessage = '数据加密失败',
    super.originalError,
  }) : super(code: 'ENCRYPT_ERROR');
}

// ─── 网络失败 ────────────────────────────────────

final class NetworkFailure extends AppFailure {
  const NetworkFailure({
    required super.message,
    super.userMessage = '网络连接出现问题，请检查网络后重试',
    super.originalError,
  }) : super(code: 'NETWORK_ERROR');
}

final class ServerFailure extends AppFailure {
  const ServerFailure({
    required super.message,
    super.userMessage = '服务器暂时不可用，请稍后重试',
    super.originalError,
  }) : super(code: 'SERVER_ERROR');
}

final class AiServiceFailure extends AppFailure {
  const AiServiceFailure({
    required super.message,
    super.userMessage = 'AI 助手暂时不在，请稍后再来聊聊',
    super.originalError,
  }) : super(code: 'AI_SERVICE_ERROR');
}

// ─── 业务失败 ────────────────────────────────────

final class ConversationLimitFailure extends AppFailure {
  const ConversationLimitFailure({
    super.message = '免费对话次数已用完',
    super.userMessage = '你已用完 2 次免费对话，订阅月付会员继续探索吧 ✨',
  }) : super(code: 'CONVERSATION_LIMIT');
}

final class PaymentFailure extends AppFailure {
  const PaymentFailure({
    required super.message,
    super.userMessage = '支付遇到问题，请稍后重试',
    super.originalError,
  }) : super(code: 'PAYMENT_ERROR');
}

final class ValidationFailure extends AppFailure {
  const ValidationFailure({
    required super.message,
    required super.userMessage,
  }) : super(code: 'VALIDATION_ERROR');
}
