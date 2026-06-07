import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../../features/onboarding/models/test_result.dart';
import '../../features/onboarding/presentation/result_screen.dart';
import '../../features/onboarding/presentation/test_screen.dart';

/// 应用路由配置
///
/// 使用 GoRouter 实现声明式路由。
/// 路由命名遵循 feature/screen 模式。
class AppRouter {
  late final GoRouter router = GoRouter(
    initialLocation: '/initial-test',
    routes: [
      // 主框架（底部导航）
      ShellRoute(
        builder: (context, state, child) {
          // TODO Sprint 3: 实现 AppScaffold 带底部导航栏
          return Scaffold(body: child);
        },
        routes: [
          GoRoute(
            path: '/mountain',
            name: 'mountain',
            pageBuilder: (context, state) => const NoTransitionPage(
              child: Scaffold(body: Center(child: Text('🏔️ 我之山'))),
            ),
          ),
          GoRoute(
            path: '/decision',
            name: 'decision',
            pageBuilder: (context, state) => const NoTransitionPage(
              child: Scaffold(body: Center(child: Text('💭 决策助手'))),
            ),
          ),
          GoRoute(
            path: '/profile',
            name: 'profile',
            pageBuilder: (context, state) => const NoTransitionPage(
              child: Scaffold(body: Center(child: Text('👤 我的'))),
            ),
          ),
        ],
      ),

      // 全屏路由
      GoRoute(
        path: '/initial-test',
        name: 'initial-test',
        builder: (context, state) => const InitialTestScreen(),
      ),
      GoRoute(
        path: '/onboarding/result',
        name: 'onboarding-result',
        builder: (context, state) {
          final result = state.extra as TestResult;
          return TestResultScreen(result: result);
        },
      ),
      GoRoute(
        path: '/conversation',
        name: 'conversation',
        builder: (context, state) => const Scaffold(
          body: Center(child: Text('对话中...')),
        ),
      ),
      GoRoute(
        path: '/conversation/:id/report',
        name: 'decision-report',
        builder: (context, state) => const Scaffold(
          body: Center(child: Text('决策报告')),
        ),
      ),
      GoRoute(
        path: '/onboarding',
        name: 'onboarding',
        builder: (context, state) => const Scaffold(
          body: Center(child: Text('首次引导')),
        ),
      ),
      GoRoute(
        path: '/paywall',
        name: 'paywall',
        builder: (context, state) => const Scaffold(
          body: Center(child: Text('付费墙')),
        ),
      ),
      GoRoute(
        path: '/timeline',
        name: 'timeline',
        builder: (context, state) => const Scaffold(
          body: Center(child: Text('时间轴')),
        ),
      ),
      GoRoute(
        path: '/achievements',
        name: 'achievements',
        builder: (context, state) => const Scaffold(
          body: Center(child: Text('成就')),
        ),
      ),
      GoRoute(
        path: '/subpeak/:id',
        name: 'subpeak-detail',
        builder: (context, state) => const Scaffold(
          body: Center(child: Text('副峰详情')),
        ),
      ),
      GoRoute(
        path: '/decoration-shop',
        name: 'decoration-shop',
        builder: (context, state) => const Scaffold(
          body: Center(child: Text('装饰品商店')),
        ),
      ),
    ],
    errorBuilder: (context, state) => Scaffold(
      body: Center(
        child: Text('页面找不到了: ${state.error}'),
      ),
    ),
  );
}
