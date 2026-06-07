import 'package:equatable/equatable.dart';

/// 情绪标签
///
/// 借鉴 ombre_brain 的 Russell 环形模型（valence-arousal 坐标系）
/// 为每段对话标记情绪状态，驱动对话的情感化设计。
///
/// valence:  效价 0.0~1.0（消极→积极）
/// arousal:  唤醒度 0.0~1.0（平静→激动）
class EmotionTag extends Equatable {
  /// 效价（愉快度）：0.0 消极 → 1.0 积极
  final double valence;

  /// 唤醒度：0.0 平静 → 1.0 激动
  final double arousal;

  /// 主导情绪类别
  final EmotionCategory category;

  const EmotionTag({
    required this.valence,
    required this.arousal,
    required this.category,
  });

  /// 从 valence/arousal 坐标判定情绪类别
  factory EmotionTag.fromCoordinates(double valence, double arousal) {
    return EmotionTag(
      valence: valence.clamp(0.0, 1.0),
      arousal: arousal.clamp(0.0, 1.0),
      category: _classifyEmotion(valence, arousal),
    );
  }

  static EmotionCategory _classifyEmotion(double v, double a) {
    if (v >= 0.6 && a >= 0.6) return EmotionCategory.excited;
    if (v >= 0.6 && a < 0.6) return EmotionCategory.calm;
    if (v < 0.4 && a >= 0.6) return EmotionCategory.anxious;
    if (v < 0.4 && a < 0.4) return EmotionCategory.depressed;
    if (v < 0.5 && a >= 0.4) return EmotionCategory.tense;
    return EmotionCategory.neutral;
  }

  /// 情绪中文标签
  String get label => category.label;

  /// 匹配的情绪响应策略
  String get responseStrategy => category.responseStrategy;

  @override
  List<Object?> get props => [valence, arousal, category];
}

/// 情绪类别
enum EmotionCategory {
  excited('兴奋', '与用户一起庆祝，强化积极体验'),
  calm('平静', '温和陪伴，给用户安全空间'),
  anxious('焦虑', '先接纳情绪再理性引导'),
  depressed('低落', '无条件支持，传递"我在这里"'),
  tense('紧张', '帮助用户将紧张能量转化为行动'),
  neutral('中性', '保持温暖中立，等待用户引领方向');

  final String label;
  final String responseStrategy;

  const EmotionCategory(this.label, this.responseStrategy);
}
