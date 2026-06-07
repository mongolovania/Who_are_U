import 'package:flutter/material.dart';

import '../../../../shared/theme/app_colors.dart';

/// 测试进度指示器
///
/// 10 个圆形指示点，显示当前进度。
/// - 已完成：暖橙实心
/// - 当前题：暖橙边框 + 放大
/// - 未完成：灰色半透明
class TestProgressIndicator extends StatelessWidget {
  final int total;
  final int current;
  final Set<int> completed;

  const TestProgressIndicator({
    super.key,
    required this.total,
    required this.current,
    required this.completed,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 32,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: List.generate(total, (i) {
          final isCurrent = i == current;
          final isCompleted = completed.contains(i);

          return AnimatedContainer(
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeOutCubic,
            width: isCurrent ? 28 : 10,
            height: 10,
            margin: const EdgeInsets.symmetric(horizontal: 3),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(5),
              color: isCompleted || isCurrent
                  ? AppColors.primary
                  : AppColors.textSecondary.withAlpha(40),
            ),
          );
        }),
      ),
    );
  }
}
