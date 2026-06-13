import 'package:equatable/equatable.dart';

/// 大师领域枚举
enum Domain {
  economy,          // 经济/商业
  lifePhilosophy,   // 人生哲学
  loveRelationship, // 爱情/关系
  career;           // 职场/事业

  String get label {
    switch (this) {
      case Domain.economy:
        return '经济/商业';
      case Domain.lifePhilosophy:
        return '人生哲学';
      case Domain.loveRelationship:
        return '爱情/关系';
      case Domain.career:
        return '职场/事业';
    }
  }

  String get emoji {
    switch (this) {
      case Domain.economy:
        return '📈';
      case Domain.lifePhilosophy:
        return '🏛';
      case Domain.loveRelationship:
        return '💕';
      case Domain.career:
        return '💼';
    }
  }
}

/// 大师模型
///
/// 每位大师有领域归属、方法论摘要、金句和使用指南。
/// 数据存储在 assets/masters.json。
class Master extends Equatable {
  final String id;
  final String nameCn;
  final String nameEn;
  final Domain domain;
  final String methodology;  // ~200字方法论摘要
  final String goldenQuote;  // 金句
  final String usageGuide;   // 使用指南（何时引用、如何引用）
  final List<String> tags;   // 关键词标签

  const Master({
    required this.id,
    required this.nameCn,
    required this.nameEn,
    required this.domain,
    required this.methodology,
    required this.goldenQuote,
    required this.usageGuide,
    this.tags = const [],
  });

  factory Master.fromJson(Map<String, dynamic> json) {
    return Master(
      id: json['id'] as String,
      nameCn: json['nameCn'] as String,
      nameEn: json['nameEn'] as String,
      domain: Domain.values.firstWhere(
        (d) => d.name == json['domain'],
        orElse: () => Domain.lifePhilosophy,
      ),
      methodology: json['methodology'] as String? ?? '',
      goldenQuote: json['goldenQuote'] as String? ?? '',
      usageGuide: json['usageGuide'] as String? ?? '',
      tags: (json['tags'] as List<dynamic>?)
              ?.map((e) => e as String)
              .toList() ??
          const [],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'nameCn': nameCn,
      'nameEn': nameEn,
      'domain': domain.name,
      'methodology': methodology,
      'goldenQuote': goldenQuote,
      'usageGuide': usageGuide,
      'tags': tags,
    };
  }

  @override
  List<Object?> get props => [id, nameCn, nameEn, domain];
}
