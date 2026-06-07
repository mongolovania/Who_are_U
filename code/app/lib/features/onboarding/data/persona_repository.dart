import 'dart:convert';

import 'package:logger/logger.dart';

import '../../../core/storage/database.dart';
import '../../../shared/models/user.dart';

/// 用户画像仓储
///
/// 管理 user_profile 表的读写操作。
/// user_profile 是单行表 (id=1)，设备即用户。
class PersonaRepository {
  final AppDatabase _db;
  final Logger _logger = Logger('PersonaRepository');

  PersonaRepository({required AppDatabase db}) : _db = db;

  /// 保存用户画像特质
  ///
  /// 替换整个 traits_json 列。
  /// 如果单行不存在则插入，否则更新。
  Future<void> saveTraits(List<PersonaTrait> traits) async {
    final now = DateTime.now().millisecondsSinceEpoch;
    final traitsJson = jsonEncode(traits.map((t) => t.toJson()).toList());

    final existing = await _db.db.query('user_profile', where: 'id = 1');

    if (existing.isEmpty) {
      await _db.db.insert('user_profile', {
        'id': 1,
        'traits_json': traitsJson,
        'created_at': now,
        'updated_at': now,
      });
    } else {
      await _db.db.update(
        'user_profile',
        {'traits_json': traitsJson, 'updated_at': now},
        where: 'id = 1',
      );
    }

    _logger.i('画像已保存: ${traits.map((t) => t.label).join(', ')}');
  }

  /// 读取用户画像特质
  ///
  /// 返回当前保存的特质列表，如果没有则返回空列表。
  Future<List<PersonaTrait>> loadTraits() async {
    final rows = await _db.db.query('user_profile', where: 'id = 1');

    if (rows.isEmpty) return [];

    final traitsJson = rows.first['traits_json'] as String;
    if (traitsJson.isEmpty || traitsJson == '[]') return [];

    final List<dynamic> jsonList = jsonDecode(traitsJson) as List<dynamic>;
    return jsonList
        .map((j) => PersonaTrait.fromJson(j as Map<String, dynamic>))
        .toList();
  }

  /// 检查是否已完成初始测试
  ///
  /// 如果 user_profile 表中存在 traits 数据则返回 true。
  Future<bool> hasCompletedTest() async {
    final traits = await loadTraits();
    return traits.isNotEmpty;
  }
}
