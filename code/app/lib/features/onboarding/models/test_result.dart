import 'package:equatable/equatable.dart';

import '../../../shared/models/user.dart';
import '../data/scoring_engine.dart';

/// 初始测试结果
///
/// 包含用户对 10 题的原始回答和计算出的 3 个特质泡泡。
class TestResult extends Equatable {
  /// questionIndex → score (1-5)
  final Map<int, int> answers;

  /// 计算出的 3 个核心特质
  final List<PersonaTrait> traits;

  /// 完成时间
  final DateTime completedAt;

  const TestResult({
    required this.answers,
    required this.traits,
    required this.completedAt,
  });

  /// 从原始回答计算测试结果
  ///
  /// 要求 [answers] 恰好包含 10 个回答 (index 0-9)。
  /// 内部调用 [ScoringEngine.calculate] 计算特质。
  factory TestResult.fromAnswers(Map<int, int> answers) {
    if (answers.length != 10) {
      throw ArgumentError('测试需要恰好 10 个回答，收到 ${answers.length} 个');
    }
    for (final score in answers.values) {
      if (score < 1 || score > 5) {
        throw ArgumentError('每题分数需在 1-5 之间，收到 $score');
      }
    }

    final traits = ScoringEngine.calculate(answers);

    return TestResult(
      answers: Map.unmodifiable(answers),
      traits: traits,
      completedAt: DateTime.now(),
    );
  }

  @override
  List<Object?> get props => [answers, traits, completedAt];
}
