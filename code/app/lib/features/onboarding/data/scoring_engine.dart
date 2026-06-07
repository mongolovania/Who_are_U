import '../../../shared/models/user.dart';
import '../models/test_question.dart';
import 'test_questions.dart';

/// 评分引擎
///
/// 将 10 题 Likert 回答映射为 3 个 [PersonaTrait]。
///
/// ## 评分逻辑
///
/// 每个维度 3 道专属题 + Q10 混合题 = 4 个数据点。
/// 每题 1-5 分 → 维度平均分 / 5 = trait.score (0.0-1.0)。
///
/// 维度分 >= 3.0 → 高分 trait（冒险家/点子王/共情者）
/// 维度分 < 3.0  → 低分 trait（守护者/坚韧者/完美主义者）
class ScoringEngine {
  ScoringEngine._();

  /// 6 个特质的完整定义
  static const Map<String, _TraitDefinition> _traits = {
    'adventurer': _TraitDefinition(
      key: 'adventurer',
      label: '冒险家',
      emoji: '🧗',
      description: '你渴望探索未知，愿意承担风险去追寻新的可能。'
          '你的勇气让你看到别人看不到的风景——但也别忘了偶尔回头看看脚下的路。',
    ),
    'guardian': _TraitDefinition(
      key: 'guardian',
      label: '守护者',
      emoji: '🛡️',
      description: '你重视安全与稳定，在风浪中是你为身边的人撑起保护伞。'
          '你的审慎不是保守，而是一种深沉的力量——可别把自己锁得太紧。',
    ),
    'idea_king': _TraitDefinition(
      key: 'idea_king',
      label: '点子王',
      emoji: '💡',
      description: '你的大脑像一个永不熄灭的灯泡工厂，总能源源不断产生新想法。'
          '发散是你的天赋——偶尔聚焦一下，你会发现更深的东西。',
    ),
    'resilient': _TraitDefinition(
      key: 'resilient',
      label: '坚韧者',
      emoji: '🪨',
      description: '你能承受大多数人心碎的压力，像山一样稳稳地站在那里。'
          '你的韧性让人敬佩——也请允许自己有脆弱的时刻，那不是软弱。',
    ),
    'empath': _TraitDefinition(
      key: 'empath',
      label: '共情者',
      emoji: '💛',
      description: '你天生能感受到他人的情绪，像一面温暖的镜子映照着周围的世界。'
          '这份敏感是珍贵的礼物——记得把温柔也留一些给自己。',
    ),
    'perfectionist': _TraitDefinition(
      key: 'perfectionist',
      label: '完美主义者',
      emoji: '✨',
      description: '你追求卓越，对事物有着近乎执着的标准。'
          '你的严谨让每一件事都闪闪发光——偶尔允许自己"差不多就好"，那也很美。',
    ),
  };

  /// 根据 10 题答案计算 3 个 PersonaTrait
  ///
  /// [answers] key=questionIndex(0-9), value=score(1-5)
  /// 返回 3 个按 score 降序排列的 PersonaTrait。
  static List<PersonaTrait> calculate(Map<int, int> answers) {
    final results = <PersonaTrait>[];

    for (final dim in allDimensions) {
      final score = _dimensionScore(answers, dim);
      final traitKey = _traitForDimension(dim, score);
      final def = _traits[traitKey]!;

      results.add(PersonaTrait(
        key: def.key,
        label: def.label,
        emoji: def.emoji,
        description: def.description,
        score: (score / 5.0).clamp(0.0, 1.0),
      ));
    }

    // 按 score 降序排列
    results.sort((a, b) => b.score.compareTo(a.score));
    return results;
  }

  /// 计算单个维度的平均得分（0.0-5.0）
  static double _dimensionScore(
    Map<int, int> answers,
    PersonalityDimension dim,
  ) {
    // 获取该维度的所有题目
    final dimQuestions = testQuestions
        .where((q) => q.dimension == dim)
        .toList();

    // Q10 (index=9) 计入所有维度
    final q10Score = answers[9] ?? 3;

    // 维度专属题得分
    final dimScores = dimQuestions
        .where((q) => q.index != 9) // 排除 Q10（它已经在 dimQuestions 中）
        .map((q) => answers[q.index] ?? 3)
        .toList();

    // 如果维度没有专属题，只用 Q10
    if (dimScores.isEmpty) {
      return q10Score.toDouble();
    }

    // 维度平均 = (专属题平均 + Q10) / 2
    // 但专属题有 3 道，权重更高
    final dimAvg = dimScores.reduce((a, b) => a + b) / dimScores.length;
    return (dimAvg * 3 + q10Score) / 4; // 专属题权重 75%, Q10 权重 25%
  }

  /// 根据维度得分选择对应的特质
  ///
  /// 得分 >= 3.0 → 高分 trait
  /// 得分 < 3.0  → 低分 trait
  static String _traitForDimension(PersonalityDimension dim, double score) {
    final isHigh = score >= 3.0;

    return switch (dim) {
      PersonalityDimension.action =>
        isHigh ? 'adventurer' : 'guardian',
      PersonalityDimension.thinking =>
        isHigh ? 'idea_king' : 'resilient',
      PersonalityDimension.emotion =>
        isHigh ? 'empath' : 'perfectionist',
    };
  }
}

/// 特质定义（内部使用）
class _TraitDefinition {
  final String key;
  final String label;
  final String emoji;
  final String description;

  const _TraitDefinition({
    required this.key,
    required this.label,
    required this.emoji,
    required this.description,
  });
}
