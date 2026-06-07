import 'package:flutter/material.dart';

import '../../../../shared/theme/app_colors.dart';

/// Likert 5 点量表选择器
///
/// 水平排列 5 个圆形按钮，用户点击选择同意程度。
/// 选中状态：暖橙填充 + 微缩放动画。
class LikertScale extends StatelessWidget {
  /// 当前选中的分数（null = 未选）
  final int? selectedScore;

  /// 选中回调
  final ValueChanged<int> onSelected;

  const LikertScale({
    super.key,
    required this.selectedScore,
    required this.onSelected,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: List.generate(5, (i) {
        final score = i + 1;
        final isSelected = selectedScore == score;

        return _ScaleButton(
          score: score,
          isSelected: isSelected,
          onTap: () => onSelected(score),
        );
      }),
    );
  }
}

class _ScaleButton extends StatelessWidget {
  final int score;
  final bool isSelected;
  final VoidCallback onTap;

  const _ScaleButton({
    required this.score,
    required this.isSelected,
    required this.onTap,
  });

  static const _labels = [
    '很不\n同意',
    '不太\n同意',
    '不\n确定',
    '比较\n同意',
    '非常\n同意',
  ];

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 250),
        curve: Curves.easeOutCubic,
        width: isSelected ? 64 : 56,
        height: isSelected ? 64 : 56,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: isSelected ? AppColors.primary : AppColors.surface,
          boxShadow: isSelected
              ? [
                  BoxShadow(
                    color: AppColors.primary.withAlpha(80),
                    blurRadius: 12,
                    offset: const Offset(0, 4),
                  ),
                ]
              : [
                  BoxShadow(
                    color: Colors.black.withAlpha(10),
                    blurRadius: 4,
                    offset: const Offset(0, 2),
                  ),
                ],
          border: Border.all(
            color: isSelected
                ? AppColors.primary
                : AppColors.textSecondary.withAlpha(40),
            width: isSelected ? 2 : 1,
          ),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              '$score',
              style: TextStyle(
                fontSize: isSelected ? 20 : 16,
                fontWeight: isSelected ? FontWeight.w700 : FontWeight.w500,
                color: isSelected
                    ? AppColors.textOnPrimary
                    : AppColors.textSecondary,
              ),
            ),
            if (isSelected)
              Padding(
                padding: const EdgeInsets.only(top: 1),
                child: Text(
                  _labels[score - 1],
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    fontSize: 9,
                    fontWeight: FontWeight.w500,
                    color: AppColors.textOnPrimary,
                    height: 1.1,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
