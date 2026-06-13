import 'dart:math';

import 'package:logger/logger.dart';

import '../../archetypes/data/archetypes.dart' show allArchetypes;
import '../../archetypes/models/archetype.dart';
import '../models/master.dart';
import 'master_registry.dart';

final _log = Logger();
final _random = Random();

/// 大师选择引擎
///
/// 24原型 × 50大师 匹配矩阵 → top 7 参谋。
///
/// 算法（来自总体技术方案 §7.2）：
///   1. 用户原型(1-3个) → 查找兼容大师候选
///   2. 决策领域 → 按领域筛选
///   3. 用户提及大师 → 权重×2 强制入选
///   4. 去重 + 评分排序 → 返回 top 7
class SelectionEngine {
  final MasterRegistry _registry;

  /// 24原型 × 大师 匹配权重矩阵
  /// archetypeId → {masterId: weight}
  final Map<String, Map<String, double>> _matchMatrix = {};

  SelectionEngine(this._registry);

  /// 构建匹配矩阵
  ///
  /// 基于标签重叠度 + 阵营-领域亲和度计算权重。
  void buildMatrix() {
    final allMasters = _registry.all;
    _matchMatrix.clear();

    // 阵营 → 领域的亲和度
    const factionDomainAffinity = {
      Faction.nurturer: {
        Domain.loveRelationship: 0.9,
        Domain.lifePhilosophy: 0.6,
        Domain.career: 0.4,
        Domain.economy: 0.2,
      },
      Faction.warrior: {
        Domain.career: 0.9,
        Domain.economy: 0.7,
        Domain.lifePhilosophy: 0.5,
        Domain.loveRelationship: 0.3,
      },
      Faction.sage: {
        Domain.lifePhilosophy: 1.0,
        Domain.economy: 0.7,
        Domain.career: 0.5,
        Domain.loveRelationship: 0.4,
      },
      Faction.creator: {
        Domain.career: 0.8,
        Domain.lifePhilosophy: 0.6,
        Domain.economy: 0.6,
        Domain.loveRelationship: 0.3,
      },
    };

    // 为每个原型计算对所有大师的权重
    for (final archetype in allArchetypes) {
      final weights = <String, double>{};
      final factionAffinity = factionDomainAffinity[archetype.faction]!;

      for (final master in allMasters) {
        double score = 0.0;

        // 1. 阵营-领域亲和度（权重 0.4）
        score += (factionAffinity[master.domain] ?? 0.3) * 0.4;

        // 2. 标签重叠度（权重 0.3）
        if (archetype.compatibleMasterIds.contains(master.id)) {
          score += 0.3; // 直接兼容标记
        }

        // 3. 原型特质与大师标签的语义关联（权重 0.3）
        // 简化版：基于原型关键词和大师标签的匹配
        final archetypeKeywords = _extractKeywords(archetype);
        final masterKeywords = master.tags.toSet();
        final overlap = archetypeKeywords.intersection(masterKeywords).length;
        score += min(overlap / max(archetypeKeywords.length, 1), 1.0) * 0.3;

        weights[master.id] = score;
      }

      _matchMatrix[archetype.id] = weights;
    }

    _log.i('Match matrix built: ${_matchMatrix.length} archetypes × ${allMasters.length} masters');
  }

  /// 提取原型关键词（用于匹配）
  Set<String> _extractKeywords(Archetype archetype) {
    final keywords = <String>{};

    // 从驱动力和一句话中提取关键词
    final text = '${archetype.drive} ${archetype.oneLiner} ${archetype.nameCn}';

    // 通用关键词
    const keywordMap = {
      '勇气': ['勇气', '冒险', '勇敢', '挑战'],
      '创造': ['创新', '创造', '发明', '突破'],
      '智慧': ['智慧', '思考', '反思', '理性'],
      '关怀': ['共情', '关爱', '治愈', '滋养'],
      '守护': ['守护', '保护', '捍卫', '安全'],
      '探索': ['探索', '未知', '发现', '好奇'],
      '独立': ['独立', '自由', '个性', '自主'],
      '秩序': ['秩序', '构建', '系统', '结构'],
      '直觉': ['直觉', '感受', '预感', '内心'],
      '耐心': ['耐心', '坚持', '长期', '等待'],
      '变革': ['变革', '颠覆', '改变', '革命'],
      '务实': ['务实', '实干', '踏实', '实际'],
    };

    for (final entry in keywordMap.entries) {
      for (final kw in entry.value) {
        if (text.contains(kw)) {
          keywords.add(entry.key);
        }
      }
    }

    return keywords;
  }

  /// 选择 top 7 大师
  ///
  /// [archetypes] 用户的 1-3 个原型
  /// [domain] 决策领域（可选）
  /// [userMentions] 用户在对话中提到的大师名/ID（可选）
  ///
  /// 返回 top 7 Master 列表。
  List<Master> select({
    required List<Archetype> archetypes,
    Domain? domain,
    List<String>? userMentions,
  }) {
    if (archetypes.isEmpty) {
      _log.w('No archetypes provided, returning random masters');
      return _randomMasters(7);
    }

    // 如果矩阵未构建，直接返回领域筛选结果
    if (_matchMatrix.isEmpty) {
      _log.w('Match matrix not built, using domain-only selection');
      if (domain != null) {
        final candidates = _registry.getByDomain(domain);
        candidates.shuffle(_random);
        return candidates.take(7).toList();
      }
      return _randomMasters(7);
    }

    final scores = <String, double>{};

    // Step 1: 对每个用户原型，累加匹配权重
    for (final archetype in archetypes) {
      final weights = _matchMatrix[archetype.id];
      if (weights == null) continue;

      for (final entry in weights.entries) {
        scores[entry.key] = (scores[entry.key] ?? 0) + entry.value;
      }
    }

    // Step 2: 领域筛选 → 同领域大师加分
    if (domain != null) {
      final domainMasters = _registry.getByDomain(domain);
      for (final master in domainMasters) {
        scores[master.id] = (scores[master.id] ?? 0.3) + 0.5;
      }
    }

    // Step 3: 用户提及大师 → 权重×2 强制入选
    final forcedMasters = <Master>[];
    if (userMentions != null && userMentions.isNotEmpty) {
      for (final mention in userMentions) {
        // 尝试按 ID 和名称查找
        Master? found;
        found = _registry.getById(mention.toLowerCase());
        if (found == null) {
          // 按中文名模糊匹配
          for (final m in _registry.all) {
            if (m.nameCn.contains(mention) || m.nameEn.toLowerCase().contains(mention.toLowerCase())) {
              found = m;
              break;
            }
          }
        }
        if (found != null && !forcedMasters.contains(found)) {
          forcedMasters.add(found);
          scores[found.id] = (scores[found.id] ?? 0.5) * 2.0;
        }
      }
    }

    // Step 4: 去重 + 排序
    final ranked = scores.entries
        .where((e) => _registry.getById(e.key) != null)
        .toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    // 构建结果：强制入选大师优先，然后按评分
    final result = <Master>[];
    final seen = <String>{};

    // 先加入强制大师
    for (final m in forcedMasters) {
      if (seen.add(m.id)) {
        result.add(m);
      }
    }

    // 补足到 top 7
    for (final entry in ranked) {
      if (result.length >= 7) break;
      if (seen.add(entry.key)) {
        final master = _registry.getById(entry.key);
        if (master != null) {
          result.add(master);
        }
      }
    }

    // 如果还不够7个，随机补充
    if (result.length < 7) {
      final remaining = _registry.all.where((m) => !seen.contains(m.id)).toList();
      remaining.shuffle(_random);
      for (final m in remaining) {
        if (result.length >= 7) break;
        result.add(m);
      }
    }

    _log.i(
      'Selection: archetypes=${archetypes.map((a) => a.id).join(',')} '
      'domain=${domain?.label ?? 'none'} → '
      'top7=${result.map((m) => m.nameCn).join(', ')}',
    );

    return result.take(7).toList();
  }

  /// 随机选择 N 位大师（降级方案）
  List<Master> _randomMasters(int n) {
    final all = _registry.all.toList();
    all.shuffle(_random);
    return all.take(n).toList();
  }
}
