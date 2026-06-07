import 'package:equatable/equatable.dart';

import '../../models/test_question.dart';
import '../../models/test_result.dart';

/// 初始测试状态
///
/// 管理 10 题测试的完整生命周期：
/// answering → calculating → done
enum TestStatus { answering, calculating, done }

class TestState extends Equatable {
  final List<TestQuestion> questions;
  final int currentIndex;
  final Map<int, int> answers;
  final TestStatus status;
  final TestResult? result;
  final String? error;

  const TestState({
    required this.questions,
    this.currentIndex = 0,
    this.answers = const {},
    this.status = TestStatus.answering,
    this.result,
    this.error,
  });

  /// 当前题目
  TestQuestion get currentQuestion => questions[currentIndex];

  /// 所有题目是否都已回答
  bool get allAnswered => answers.length == questions.length;

  /// 当前题目是否已选择
  bool get currentAnswered => answers.containsKey(currentIndex);

  /// 当前题目已选分数（null 表示未选）
  int? get currentScore => answers[currentIndex];

  /// 已完成题目数
  int get completedCount => answers.length;

  TestState copyWith({
    List<TestQuestion>? questions,
    int? currentIndex,
    Map<int, int>? answers,
    TestStatus? status,
    TestResult? result,
    String? error,
  }) {
    return TestState(
      questions: questions ?? this.questions,
      currentIndex: currentIndex ?? this.currentIndex,
      answers: answers ?? this.answers,
      status: status ?? this.status,
      result: result ?? this.result,
      error: error,
    );
  }

  @override
  List<Object?> get props => [
        questions,
        currentIndex,
        answers,
        status,
        result,
        error,
      ];
}
