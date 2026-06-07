/// 应用全局配置
///
/// 管理环境变量、API 端点、功能开关等。
/// 生产环境与调试环境使用不同配置。
class AppConfig {
  final String environment;
  final String aiProxyBaseUrl;
  final String iapVerificationUrl;
  final bool isDebug;
  final int maxFreeConversations;
  final int maxConversationRounds;
  final int maxConversationMinutes;

  const AppConfig({
    required this.environment,
    required this.aiProxyBaseUrl,
    required this.iapVerificationUrl,
    required this.isDebug,
    required this.maxFreeConversations,
    required this.maxConversationRounds,
    required this.maxConversationMinutes,
  });

  /// 从环境加载配置
  factory AppConfig.load() {
    // TODO: 从 .env 文件或编译时常量读取
    // 初期硬编码开发环境配置
    const isDebug = bool.fromEnvironment('IS_DEBUG', defaultValue: true);

    return AppConfig(
      environment: isDebug ? 'development' : 'production',
      aiProxyBaseUrl: isDebug
          ? 'http://localhost:8000/api'
          : 'https://api.whoareu.app/api',
      iapVerificationUrl: isDebug
          ? 'http://localhost:8000/api/payment/verify'
          : 'https://api.whoareu.app/api/payment/verify',
      isDebug: isDebug,
      maxFreeConversations: 2,
      maxConversationRounds: 50,
      maxConversationMinutes: 30,
    );
  }

  /// 生产环境配置
  factory AppConfig.production() {
    return AppConfig(
      environment: 'production',
      aiProxyBaseUrl: 'https://api.whoareu.app/api',
      iapVerificationUrl: 'https://api.whoareu.app/api/payment/verify',
      isDebug: false,
      maxFreeConversations: 2,
      maxConversationRounds: 50,
      maxConversationMinutes: 30,
    );
  }
}
