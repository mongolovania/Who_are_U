import 'package:equatable/equatable.dart';

/// 用户画像
///
/// v1 架构：设备即用户，无账户系统。
/// 用户仅通过本地加密数据库的 traits 标识。
/// 所有数据存储在本设备上，不上传服务器。
///
/// v2 架构规划：账户绑定后，此模型扩展到包含 E2E 密钥材料。
class User extends Equatable {
  final List<PersonaTrait> traits;
  final DateTime createdAt;
  final DateTime updatedAt;

  const User({
    this.traits = const [],
    required this.createdAt,
    required this.updatedAt,
  });

  /// 创建新用户（首次完成初始测试后）
  factory User.create(List<PersonaTrait> traits) {
    final now = DateTime.now();
    return User(
      traits: traits,
      createdAt: now,
      updatedAt: now,
    );
  }

  User copyWith({
    List<PersonaTrait>? traits,
    DateTime? createdAt,
    DateTime? updatedAt,
  }) {
    return User(
      traits: traits ?? this.traits,
      createdAt: createdAt ?? this.createdAt,
      updatedAt: updatedAt ?? this.updatedAt,
    );
  }

  @override
  List<Object?> get props => [traits, updatedAt];
}

/// 用户画像特质
///
/// 从初始 10 题测试中生成 3 个特质泡泡。
/// 对话过程中动态更新。
class PersonaTrait extends Equatable {
  final String key;
  final String label;
  final String emoji;
  final String description;
  final double score; // 0.0 ~ 1.0 强度

  const PersonaTrait({
    required this.key,
    required this.label,
    required this.emoji,
    required this.description,
    required this.score,
  });

  /// 从 JSON 反序列化
  factory PersonaTrait.fromJson(Map<String, dynamic> json) {
    return PersonaTrait(
      key: json['key'] as String,
      label: json['label'] as String,
      emoji: json['emoji'] as String? ?? '',
      description: json['description'] as String? ?? '',
      score: (json['score'] as num).toDouble(),
    );
  }

  /// 序列化为 JSON
  Map<String, dynamic> toJson() {
    return {
      'key': key,
      'label': label,
      'emoji': emoji,
      'description': description,
      'score': score,
    };
  }

  @override
  List<Object?> get props => [key, label, score];
}
