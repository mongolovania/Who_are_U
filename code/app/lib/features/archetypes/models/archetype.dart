import 'package:equatable/equatable.dart';

/// 阵营枚举 — 4大阵营
enum Faction {
  nurturer,  // 滋养者
  warrior,   // 行动者
  sage,      // 智者
  creator;   // 创变者

  String get label {
    switch (this) {
      case Faction.nurturer:
        return '滋养者';
      case Faction.warrior:
        return '行动者';
      case Faction.sage:
        return '智者';
      case Faction.creator:
        return '创变者';
    }
  }

  String get emoji {
    switch (this) {
      case Faction.nurturer:
        return '🌿';
      case Faction.warrior:
        return '⚔️';
      case Faction.sage:
        return '🔮';
      case Faction.creator:
        return '🎨';
    }
  }
}

/// 24原型模型
///
/// 每个原型属于4大阵营之一，有核心驱动力和一句话描述。
/// 与50大师通过 [compatibleMasterIds] 建立关联。
class Archetype extends Equatable {
  final String id;
  final String nameCn;
  final String nameEn;
  final String emoji;
  final Faction faction;
  final String drive; // 核心驱动力
  final String oneLiner; // 一句话
  final List<String> compatibleMasterIds;

  const Archetype({
    required this.id,
    required this.nameCn,
    required this.nameEn,
    required this.emoji,
    required this.faction,
    required this.drive,
    required this.oneLiner,
    this.compatibleMasterIds = const [],
  });

  /// 从 JSON 反序列化
  factory Archetype.fromJson(Map<String, dynamic> json) {
    return Archetype(
      id: json['id'] as String,
      nameCn: json['nameCn'] as String,
      nameEn: json['nameEn'] as String,
      emoji: json['emoji'] as String,
      faction: Faction.values.firstWhere(
        (f) => f.name == json['faction'],
        orElse: () => Faction.nurturer,
      ),
      drive: json['drive'] as String? ?? '',
      oneLiner: json['oneLiner'] as String? ?? '',
      compatibleMasterIds: (json['compatibleMasterIds'] as List<dynamic>?)
              ?.map((e) => e as String)
              .toList() ??
          const [],
    );
  }

  /// 序列化为 JSON
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'nameCn': nameCn,
      'nameEn': nameEn,
      'emoji': emoji,
      'faction': faction.name,
      'drive': drive,
      'oneLiner': oneLiner,
      'compatibleMasterIds': compatibleMasterIds,
    };
  }

  @override
  List<Object?> get props => [id, nameCn, nameEn, faction];
}
