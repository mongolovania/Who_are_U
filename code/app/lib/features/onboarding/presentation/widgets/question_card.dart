import 'package:flutter/material.dart';

import '../../../../shared/theme/app_colors.dart';
import '../../models/test_question.dart';
import 'likert_scale.dart';

/// 题目卡片
///
/// 展示单道测试题：题目编号 + 文本 + 5 点量表。
class QuestionCard extends StatelessWidget {
  final TestQuestion question;
  final int? selectedScore;
  final ValueChanged<int> onSelected;

  const QuestionCard({
    super.key,
    required this.question,
    required this.selectedScore,
    required this.onSelected,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 32),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // 题目编号
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
            decoration: BoxDecoration(
              color: AppColors.primary.withAlpha(25),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              'Q${question.index + 1}/10',
              style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: AppColors.primary,
              ),
            ),
          ),
          const SizedBox(height: 24),

          // 题目文本
          Text(
            question.text,
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                  height: 1.5,
                ),
          ),
          const SizedBox(height: 36),

          // Likert 量表
          LikertScale(
            selectedScore: selectedScore,
            onSelected: onSelected,
          ),
        ],
      ),
    );
  }
}
