import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:logger/logger.dart';

import '../../data/test_questions.dart';
import '../../models/test_result.dart';
import 'test_state.dart';

/// 初始测试状态管理
///
/// 处理 10 题测试的用户交互：
/// - 选择答案 (selectAnswer)
/// - 上/下题切换 (next/previous)
/// - 自动计算 (第 10 题选完后调用 ScoringEngine)
class TestCubit extends Cubit<TestState> {
  final Logger _logger = Logger('TestCubit');

  TestCubit() : super(TestState(questions: testQuestions));

  /// 选择当前题目的答案
  void selectAnswer(int score) {
    if (state.status != TestStatus.answering) return;

    final newAnswers = Map<int, int>.from(state.answers);
    newAnswers[state.currentIndex] = score;

    emit(state.copyWith(answers: newAnswers));
    _logger.d('Q${state.currentIndex + 1} 选择: $score');
  }

  /// 前往下一题
  ///
  /// 如果已是最后一题且全部回答完毕，自动计算。
  void next() {
    if (state.currentIndex >= state.questions.length - 1) {
      // 最后一题 → 计算结果
      if (state.allAnswered) {
        _calculate();
      }
      return;
    }

    emit(state.copyWith(currentIndex: state.currentIndex + 1));
  }

  /// 返回上一题
  void previous() {
    if (state.currentIndex > 0) {
      emit(state.copyWith(currentIndex: state.currentIndex - 1));
    }
  }

  /// 跳转到指定题目
  void jumpTo(int index) {
    if (index >= 0 && index < state.questions.length) {
      emit(state.copyWith(currentIndex: index));
    }
  }

  /// 直接计算结果（用户点击"查看结果"按钮）
  void finish() {
    if (state.allAnswered) {
      _calculate();
    }
  }

  void _calculate() {
    try {
      emit(state.copyWith(status: TestStatus.calculating));

      final result = TestResult.fromAnswers(state.answers);

      emit(state.copyWith(
        status: TestStatus.done,
        result: result,
      ));

      _logger.i('测试完成 — 特质: ${result.traits.map((t) => t.label).join(', ')}');
    } catch (e) {
      emit(state.copyWith(
        status: TestStatus.answering,
        error: '计算失败，请重试: $e',
      ));
      _logger.e('评分计算失败', error: e);
    }
  }
}
