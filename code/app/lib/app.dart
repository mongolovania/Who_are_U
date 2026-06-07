import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import 'core/config/app_config.dart';
import 'core/config/theme_config.dart';
import 'core/router/app_router.dart';
import 'shared/theme/app_theme.dart';

/// 你谁啊 App 根组件
///
/// 提供全局 MaterialApp 配置，包括：
/// - 温暖色调主题
/// - GoRouter 路由
/// - 全局错误处理
class WhoAreUApp extends StatelessWidget {
  const WhoAreUApp({super.key});

  @override
  Widget build(BuildContext context) {
    final appConfig = context.read<AppConfig>();
    final appRouter = AppRouter();

    return MaterialApp.router(
      title: '你谁啊',
      debugShowCheckedModeBanner: appConfig.isDebug,

      // 主题
      theme: AppTheme.lightTheme,
      darkTheme: AppTheme.darkTheme,
      themeMode: ThemeMode.light, // 初期仅支持亮色

      // 路由
      routerConfig: appRouter.router,

      // 本地化
      locale: const Locale('zh', 'CN'),
      supportedLocales: const [
        Locale('zh', 'CN'),
      ],
      localizationsDelegates: const [
        DefaultMaterialLocalizations.delegate,
        DefaultWidgetsLocalizations.delegate,
      ],

      // 全局 Builder
      builder: (context, child) {
        return MediaQuery(
          data: MediaQuery.of(context).copyWith(
            textScaler: const TextScaler.linear(1.0),
          ),
          child: child!,
        );
      },
    );
  }
}

/// 严重错误时的降级界面
///
/// 当应用初始化失败（如数据库损坏）时展示，
/// 提示用户重启或联系支持。
class CriticalErrorApp extends StatelessWidget {
  const CriticalErrorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '你谁啊',
      home: Scaffold(
        backgroundColor: const Color(0xFF1A1A2E),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(32.0),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(
                  Icons.cloud_off_rounded,
                  size: 64,
                  color: Colors.white54,
                ),
                const SizedBox(height: 24),
                Text(
                  '出了点小问题',
                  style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                        color: Colors.white,
                      ),
                ),
                const SizedBox(height: 12),
                const Text(
                  '请完全关闭 App 后重新打开。\n如果问题持续，请联系我们。',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Colors.white54),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
