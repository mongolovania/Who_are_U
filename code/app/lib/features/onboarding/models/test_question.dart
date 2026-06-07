import 'package:equatable/equatable.dart';

/// 人格维度
///
/// 3 个双极维度，每个维度映射一对特质：
/// - action (行动倾向): 冒险家 ↔ 守护者
/// - thinking (思维模式): 点子王 ↔ 坚韧者
/// - emotion (情感倾向): 共情者 ↔ 完美主义者
enum PersonalityDimension {
  /// 行动倾向 — 面对变化和风险的态度
  action,

  /// 思维模式 — 处理问题和做计划的方式
  thinking,

  /// 情感倾向 — 做决定时的情感参考系
  emotion,
}

/// 初始测试题目
///
/// 10 道 Likert 量表题，用户选择 1-5 同意程度。
/// 每题映射到一个 [PersonalityDimension]。
class TestQuestion extends Equatable {
  /// 题目标识（q1~q10）
  final String id;

  /// 题目序号（0-9）
  final int index;

  /// 题目文本
  final String text;

  /// 所属人格维度
  final PersonalityDimension dimension;

  const TestQuestion({
    required this.id,
    required this.index,
    required this.text,
    required this.dimension,
  });

  @override
  List<Object?> get props => [id, index, text, dimension];
}
