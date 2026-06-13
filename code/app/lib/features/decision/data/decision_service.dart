import 'package:logger/logger.dart';

import '../../../shared/models/user.dart';
import '../../archetypes/data/archetype_registry.dart';
import '../../archetypes/models/archetype.dart';
import '../../masters/data/master_registry.dart';
import '../../masters/data/selection_engine.dart';
import '../../masters/models/master.dart';

final _log = Logger();

/// 决策智囊门面
///
/// 统一入口，串联 ArchetypeRegistry + MasterRegistry + SelectionEngine。
/// Sprint 4 对话引擎通过此门面获取用户原型和大师推荐。
class DecisionService {
  final ArchetypeRegistry _archetypeRegistry;
  final MasterRegistry _masterRegistry;
  late final SelectionEngine _selectionEngine;

  bool _initialized = false;

  DecisionService({
    ArchetypeRegistry? archetypeRegistry,
    MasterRegistry? masterRegistry,
  })  : _archetypeRegistry = archetypeRegistry ?? ArchetypeRegistry(),
        _masterRegistry = masterRegistry ?? MasterRegistry();

  /// 是否已初始化
  bool get isInitialized => _initialized;

  /// 初始化：加载大师数据 + 构建匹配矩阵
  Future<void> initialize() async {
    if (_initialized) return;

    await _masterRegistry.load();
    _selectionEngine = SelectionEngine(_masterRegistry);
    _selectionEngine.buildMatrix();

    _initialized = true;
    _log.i('DecisionService initialized: '
        '${_archetypeRegistry.all.length} archetypes, '
        '${_masterRegistry.all.length} masters');
  }

  /// 获取用户的 1-3 个原型
  List<Archetype> getUserArchetypes(User user) {
    return _archetypeRegistry.fromTraits(user.traits);
  }

  /// 选择 top 7 大师
  ///
  /// [archetypes] 用户原型
  /// [domain] 决策领域（可选，null 则不过滤领域）
  /// [userMentions] 用户在对话中提到的大师名
  List<Master> selectMasters({
    required List<Archetype> archetypes,
    Domain? domain,
    List<String>? userMentions,
  }) {
    if (!_initialized) {
      _log.w('DecisionService not initialized, returning empty');
      return [];
    }
    return _selectionEngine.select(
      archetypes: archetypes,
      domain: domain,
      userMentions: userMentions,
    );
  }

  /// 获取原型注册表（供外部查询）
  ArchetypeRegistry get archetypes => _archetypeRegistry;

  /// 获取大师注册表（供外部查询）
  MasterRegistry get masters => _masterRegistry;
}
