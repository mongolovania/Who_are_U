import 'package:logger/logger.dart';

import '../../../shared/models/user.dart';
import '../models/archetype.dart';
import 'archetypes.dart';
import 'trait_archetype_mapping.dart';

final _log = Logger();

/// 原型注册表
///
/// 从用户的 PersonaTrait 推导 24原型匹配。
/// 算法：
///   1. 对每个 trait 查找候选 Archetype ID
///   2. 按 trait.score 加权
///   3. 去重
///   4. 返回 top 1-3 个 Archetype
class ArchetypeRegistry {
  final Map<String, Archetype> _byId = {};

  ArchetypeRegistry() {
    for (final a in allArchetypes) {
      _byId[a.id] = a;
    }
  }

  /// 从用户特质推导原型
  ///
  /// [traits] 为 Sprint 2 测试输出的 3 个 PersonaTrait（按 score 降序）。
  /// 返回 1-3 个匹配的 Archetype。
  List<Archetype> fromTraits(List<PersonaTrait> traits) {
    if (traits.isEmpty) {
      _log.w('No traits provided, returning empty');
      return [];
    }

    // 收集候选 Archetype ID（含去重）
    final seen = <String>{};
    final candidates = <_Candidate>[];

    for (final trait in traits) {
      final archetypeIds = traitToArchetypeIds[trait.key];
      if (archetypeIds == null) {
        _log.w('Unknown trait key: ${trait.key}');
        continue;
      }

      for (final id in archetypeIds) {
        if (seen.contains(id)) continue;
        seen.add(id);

        final archetype = _byId[id];
        if (archetype == null) {
          _log.w('Unknown archetype ID: $id');
          continue;
        }

        candidates.add(_Candidate(
          archetype: archetype,
          score: trait.score, // trait.score 作为该候选的权重
        ));
      }
    }

    // 按 score 降序
    candidates.sort((a, b) => b.score.compareTo(a.score));

    // 返回 top 1-3
    final count = candidates.length.clamp(1, 3);
    final result = candidates.take(count).map((c) => c.archetype).toList();

    _log.i(
      'Traits → Archetypes: ${traits.map((t) => t.key).join(', ')} → '
      '${result.map((a) => a.id).join(', ')}',
    );

    return result;
  }

  /// 按 ID 查找原型
  Archetype? getById(String id) => _byId[id];

  /// 按阵营筛选
  List<Archetype> getByFaction(Faction faction) {
    return allArchetypes.where((a) => a.faction == faction).toList();
  }

  /// 获取全部 24 原型
  List<Archetype> get all => allArchetypes;
}

/// 内部候选结构
class _Candidate {
  final Archetype archetype;
  final double score;

  const _Candidate({required this.archetype, required this.score});
}
