import 'dart:convert';

import 'package:flutter/services.dart' show rootBundle;
import 'package:logger/logger.dart';

import '../models/master.dart';

final _log = Logger();

/// 大师注册表
///
/// 启动时从 assets/masters.json 加载50位大师数据。
/// 提供按领域、ID 查询的能力。
class MasterRegistry {
  final Map<String, Master> _byId = {};
  final Map<Domain, List<Master>> _byDomain = {};
  bool _loaded = false;

  /// 从 assets 加载大师数据
  Future<void> load() async {
    if (_loaded) return;

    try {
      final jsonStr = await rootBundle.loadString('assets/masters.json');
      _parseJson(jsonStr);
    } catch (e) {
      _log.e('Failed to load masters', error: e);
    }
  }

  /// 从 JSON 字符串加载（测试用）
  void loadFromString(String jsonStr) {
    _parseJson(jsonStr);
  }

  void _parseJson(String jsonStr) {
    final List<dynamic> data = json.decode(jsonStr) as List<dynamic>;

    for (final item in data) {
      final master = Master.fromJson(item as Map<String, dynamic>);
      _byId[master.id] = master;
      _byDomain.putIfAbsent(master.domain, () => []).add(master);
    }

    _loaded = true;
    _log.i('Loaded ${_byId.length} masters across ${_byDomain.length} domains');
  }

  /// 按 ID 查找
  Master? getById(String id) => _byId[id];

  /// 按领域获取全部大师
  List<Master> getByDomain(Domain domain) => _byDomain[domain] ?? [];

  /// 按 ID 列表批量获取
  List<Master> getByIds(List<String> ids) {
    return ids.map((id) => _byId[id]).whereType<Master>().toList();
  }

  /// 按标签搜索
  List<Master> searchByTag(String tag) {
    return _byId.values.where((m) => m.tags.contains(tag)).toList();
  }

  /// 获取全部大师
  List<Master> get all => _byId.values.toList();

  /// 是否已加载
  bool get isLoaded => _loaded;
}
