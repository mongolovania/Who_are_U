import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:go_router/go_router.dart';

import '../../../shared/theme/app_colors.dart';
import 'cubit/test_cubit.dart';
import 'cubit/test_state.dart';
import 'widgets/question_card.dart';
import 'widgets/test_progress_indicator.dart';

/// 初始测试界面
///
/// 10 题人格测试的完整交互界面。
/// 用户通过左右滑动或底部按钮切换题目，
/// 使用 5 点 Likert 量表选择同意程度。
///
/// 完成后自动跳转到 [TestResultScreen]。
class InitialTestScreen extends StatelessWidget {
  const InitialTestScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (_) => TestCubit(),
      child: const _TestView(),
    );
  }
}

class _TestView extends StatefulWidget {
  const _TestView();

  @override
  State<_TestView> createState() => _TestViewState();
}

class _TestViewState extends State<_TestView> {
  late final PageController _pageController;

  @override
  void initState() {
    super.initState();
    _pageController = PageController();
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return BlocConsumer<TestCubit, TestState>(
      listenWhen: (prev, curr) =>
          curr.status == TestStatus.done && curr.result != null,
      listener: (context, state) {
        // 跳转到结果页
        context.push('/onboarding/result', extra: state.result);
      },
      builder: (context, state) {
        return Scaffold(
          body: Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [
                  AppColors.background,
                  Color(0xFFF5EDE0), // 暖米色
                ],
              ),
            ),
            child: SafeArea(
              child: Column(
                children: [
                  const SizedBox(height: 16),

                  // 进度指示器
                  TestProgressIndicator(
                    total: state.questions.length,
                    current: state.currentIndex,
                    completed: state.answers.keys.toSet(),
                  ),
                  const SizedBox(height: 8),

                  // 进度文字
                  Text(
                    '${state.currentIndex + 1} / ${state.questions.length}',
                    style: const TextStyle(
                      fontSize: 13,
                      color: AppColors.textSecondary,
                    ),
                  ),
                  const SizedBox(height: 24),

                  // 题目滑动区
                  Expanded(
                    child: PageView.builder(
                      controller: _pageController,
                      itemCount: state.questions.length,
                      onPageChanged: (index) {
                        context.read<TestCubit>().jumpTo(index);
                      },
                      itemBuilder: (context, index) {
                        final question = state.questions[index];
                        return AnimatedSwitcher(
                          duration: const Duration(milliseconds: 350),
                          child: QuestionCard(
                            key: ValueKey('q_${question.id}'),
                            question: question,
                            selectedScore: state.answers[index],
                            onSelected: (score) {
                              context.read<TestCubit>().selectAnswer(score);
                            },
                          ),
                        );
                      },
                    ),
                  ),
                  const SizedBox(height: 16),

                  // 底部导航
                  _BottomNav(
                    currentIndex: state.currentIndex,
                    total: state.questions.length,
                    hasSelection: state.currentAnswered,
                    allAnswered: state.allAnswered,
                    status: state.status,
                    onPrevious: () {
                      context.read<TestCubit>().previous();
                      _pageController.previousPage(
                        duration: const Duration(milliseconds: 300),
                        curve: Curves.easeOutCubic,
                      );
                    },
                    onNext: () {
                      if (state.currentIndex < state.questions.length - 1) {
                        context.read<TestCubit>().next();
                        _pageController.nextPage(
                          duration: const Duration(milliseconds: 300),
                          curve: Curves.easeOutCubic,
                        );
                      } else {
                        context.read<TestCubit>().finish();
                      }
                    },
                  ),
                  const SizedBox(height: 16),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

/// 底部导航栏
///
/// 包含"上一题""下一题/查看结果"按钮和跳题圆点。
class _BottomNav extends StatelessWidget {
  final int currentIndex;
  final int total;
  final bool hasSelection;
  final bool allAnswered;
  final TestStatus status;
  final VoidCallback onPrevious;
  final VoidCallback onNext;

  const _BottomNav({
    required this.currentIndex,
    required this.total,
    required this.hasSelection,
    required this.allAnswered,
    required this.status,
    required this.onPrevious,
    required this.onNext,
  });

  @override
  Widget build(BuildContext context) {
    final isLast = currentIndex >= total - 1;
    final isLoading = status == TestStatus.calculating;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 32),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          // 上一题
          if (currentIndex > 0)
            TextButton.icon(
              onPressed: onPrevious,
              icon: const Icon(Icons.arrow_back_rounded, size: 18),
              label: const Text('上一题'),
              style: TextButton.styleFrom(
                foregroundColor: AppColors.textSecondary,
              ),
            )
          else
            const SizedBox(width: 88),

          // 跳题圆点（仅在有选择时显示）
          if (hasSelection || isLast)
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (currentIndex > 0)
                  _JumpDot(
                    label: '上一题',
                    onTap: onPrevious,
                  ),
                const SizedBox(width: 8),
                if (isLast && allAnswered && !isLoading)
                  _ActionButton(
                    label: '查看结果 ✨',
                    onTap: onNext,
                  )
                else if (isLast && isLoading)
                  const SizedBox(
                    width: 24,
                    height: 24,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: AppColors.primary,
                    ),
                  )
                else
                  _ActionButton(
                    label: '下一题',
                    onTap: onNext,
                  ),
              ],
            ),
        ],
      ),
    );
  }
}

class _JumpDot extends StatelessWidget {
  final String label;
  final VoidCallback onTap;

  const _JumpDot({required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: AppColors.textSecondary.withAlpha(30)),
        ),
        child: Text(
          label,
          style: const TextStyle(
            fontSize: 13,
            color: AppColors.textSecondary,
          ),
        ),
      ),
    );
  }
}

class _ActionButton extends StatelessWidget {
  final String label;
  final VoidCallback onTap;

  const _ActionButton({required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: onTap,
      style: ElevatedButton.styleFrom(
        backgroundColor: AppColors.primary,
        foregroundColor: AppColors.textOnPrimary,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
      ),
      child: Text(label),
    );
  }
}
