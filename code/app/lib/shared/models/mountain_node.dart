import 'package:equatable/equatable.dart';

/// 山脉节点类型
enum MountainNodeType {
  decision,    // 决策对话节点
  difficulty,  // 困难记录节点
  subpeak,     // 副峰里程碑节点
  achievement, // 成就节点
}

/// 困难类型（对应独影 7 种动作）
enum DifficultyType {
  choice,     // 抉择型 — 站在岔路口
  assault,    // 攻坚型 — 攀爬陡峭岩壁
  endure,     // 承受型 — 蹲下抱膝后站起
  breakthrough, // 突破型 — 纵身跃过悬崖
  farewell,   // 告别型 — 回头望向山下
  lost,       // 迷茫型 — 点燃篝火等待天亮
  rebirth,    // 重生型 — 瀑布下冲刷焕新
}

/// 困难类型对应的危险系数
extension DifficultyDanger on DifficultyType {
  int get dangerLevel => switch (this) {
        DifficultyType.choice => 2,
        DifficultyType.assault => 4,
        DifficultyType.endure => 3,
        DifficultyType.breakthrough => 5,
        DifficultyType.farewell => 2,
        DifficultyType.lost => 1,
        DifficultyType.rebirth => 4,
      };
}

/// 山脉节点实体
///
/// 代表"我之山"上的一个记录点。
/// 每次深度对话、困难记录、副峰登顶或成就解锁
/// 都会在山脉上生成一个新节点。
class MountainNode extends Equatable {
  final String id;
  final MountainNodeType type;
  final String title;
  final String? summary;
  final DifficultyType? difficultyType;
  final double position; // 在山脉上的纵向位置 (0.0 山脚 → 1.0 山顶)
  final DateTime unlockedAt;
  final String? conversationId;
  final Map<String, dynamic>? metadata;

  const MountainNode({
    required this.id,
    required this.type,
    required this.title,
    this.summary,
    this.difficultyType,
    required this.position,
    required this.unlockedAt,
    this.conversationId,
    this.metadata,
  });

  /// 从数据库行创建
  factory MountainNode.fromMap(Map<String, dynamic> map) {
    return MountainNode(
      id: map['id'] as String,
      type: MountainNodeType.values.firstWhere(
        (t) => t.name == map['type'],
      ),
      title: map['title'] as String,
      summary: map['summary'] as String?,
      difficultyType: map['difficulty_type'] != null
          ? DifficultyType.values.firstWhere(
              (d) => d.name == map['difficulty_type'],
            )
          : null,
      position: (map['position'] as num).toDouble(),
      unlockedAt: DateTime.fromMillisecondsSinceEpoch(map['unlocked_at'] as int),
      conversationId: map['conversation_id'] as String?,
      metadata: map['metadata_json'] != null
          ? Map<String, dynamic>.from(
              map['metadata_json'] is String
                  ? {} // JSON decoded elsewhere
                  : map['metadata_json'],
            )
          : null,
    );
  }

  /// 转换为数据库行
  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'type': type.name,
      'title': title,
      'summary': summary,
      'difficulty_type': difficultyType?.name,
      'position': position,
      'unlocked_at': unlockedAt.millisecondsSinceEpoch,
      'conversation_id': conversationId,
      'metadata_json': metadata ?? {},
    };
  }

  /// 创建副本（不可变更新）
  MountainNode copyWith({
    String? id,
    MountainNodeType? type,
    String? title,
    String? summary,
    DifficultyType? difficultyType,
    double? position,
    DateTime? unlockedAt,
    String? conversationId,
    Map<String, dynamic>? metadata,
  }) {
    return MountainNode(
      id: id ?? this.id,
      type: type ?? this.type,
      title: title ?? this.title,
      summary: summary ?? this.summary,
      difficultyType: difficultyType ?? this.difficultyType,
      position: position ?? this.position,
      unlockedAt: unlockedAt ?? this.unlockedAt,
      conversationId: conversationId ?? this.conversationId,
      metadata: metadata ?? this.metadata,
    );
  }

  @override
  List<Object?> get props => [
        id,
        type,
        title,
        position,
        unlockedAt,
      ];
}
