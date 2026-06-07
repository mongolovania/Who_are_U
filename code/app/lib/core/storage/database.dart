import 'dart:async';

import 'package:logger/logger.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';
import 'package:sqlcipher_flutter_libs/sqlcipher_flutter_libs.dart';

import 'secure_store.dart';

/// 本地加密数据库管理器
///
/// 使用 SQLite + SQLCipher 加密存储结构化数据。
/// 管理数据库初始化、迁移和连接的完整生命周期。
///
/// 存储的表：
/// - user_profile: 用户画像特质（单行，设备即用户）
/// - mountain_nodes: 山脉节点
/// - sub_peaks: 副峰目标
/// - achievements: 成就记录
/// - decorations: 装饰品拥有
/// - purchases: 购买记录
/// - app_settings: 应用设置
class AppDatabase {
  static const String _dbName = 'who_are_u.db';
  static const int _dbVersion = 2;

  final SecureStore _secureStore;

  Database? _db;
  final Logger _logger = Logger('AppDatabase');

  /// 创建数据库管理器
  ///
  /// [secureStore] 用于获取数据库加密密钥。
  AppDatabase({required SecureStore secureStore}) : _secureStore = secureStore;

  Database get db {
    if (_db == null) {
      throw StateError('数据库未初始化，请先调用 initialize()');
    }
    return _db!;
  }

  /// 初始化数据库连接，创建表结构，执行迁移
  Future<void> initialize() async {
    final docsDir = await getApplicationDocumentsDirectory();
    final dbPath = p.join(docsDir.path, _dbName);

    _logger.i('数据库路径: $dbPath');

    _db = await openDatabase(
      dbPath,
      version: _dbVersion,
      onCreate: _onCreate,
      onUpgrade: _onUpgrade,
    );

    // 启用 SQLCipher 加密
    final encryptionKey = await _secureStore.getEncryptionKey();
    await _db!.rawQuery("PRAGMA key = '$encryptionKey'");

    _logger.i('数据库初始化完成（已加密）');
  }

  /// 创建初始表结构 (v1 → v2 由迁移处理)
  Future<void> _onCreate(Database db, int version) async {
    final batch = db.batch();

    // user_profile: 单行配置表，设备即用户
    batch.execute('''
      CREATE TABLE user_profile (
        id INTEGER PRIMARY KEY DEFAULT 1 CHECK(id = 1),
        traits_json TEXT NOT NULL DEFAULT '[]',
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      )
    ''');

    batch.execute('''
      CREATE TABLE mountain_nodes (
        id TEXT PRIMARY KEY,
        type TEXT NOT NULL CHECK(type IN ('decision', 'difficulty', 'subpeak', 'achievement')),
        title TEXT NOT NULL,
        summary TEXT,
        difficulty_type TEXT,
        position REAL NOT NULL DEFAULT 0.0,
        unlocked_at INTEGER NOT NULL,
        conversation_id TEXT,
        metadata_json TEXT DEFAULT '{}',
        created_at INTEGER NOT NULL
      )
    ''');

    batch.execute('''
      CREATE TABLE sub_peaks (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        description TEXT,
        target_date INTEGER,
        progress REAL NOT NULL DEFAULT 0.0,
        status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'completed', 'abandoned')),
        created_at INTEGER NOT NULL,
        completed_at INTEGER
      )
    ''');

    batch.execute('''
      CREATE TABLE sub_peak_milestones (
        id TEXT PRIMARY KEY,
        sub_peak_id TEXT NOT NULL REFERENCES sub_peaks(id),
        title TEXT NOT NULL,
        completed INTEGER NOT NULL DEFAULT 0,
        completed_at INTEGER,
        created_at INTEGER NOT NULL
      )
    ''');

    batch.execute('''
      CREATE TABLE achievements (
        id TEXT PRIMARY KEY,
        definition_key TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT,
        icon TEXT,
        unlocked_at INTEGER NOT NULL
      )
    ''');

    batch.execute('''
      CREATE TABLE decorations (
        id TEXT PRIMARY KEY,
        definition_key TEXT NOT NULL,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        acquired_at INTEGER NOT NULL,
        is_equipped INTEGER NOT NULL DEFAULT 0
      )
    ''');

    batch.execute('''
      CREATE TABLE purchases (
        id TEXT PRIMARY KEY,
        product_id TEXT NOT NULL,
        product_type TEXT NOT NULL CHECK(product_type IN ('monthly_subscription', 'decoration')),
        receipt TEXT,
        status TEXT NOT NULL DEFAULT 'pending',
        purchased_at INTEGER NOT NULL,
        expires_at INTEGER
      )
    ''');

    batch.execute('''
      CREATE TABLE app_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at INTEGER NOT NULL
      )
    ''');

    // 索引
    batch.execute(
        'CREATE INDEX idx_nodes_type ON mountain_nodes(type)');
    batch.execute(
        'CREATE INDEX idx_nodes_created ON mountain_nodes(created_at)');
    batch.execute(
        'CREATE INDEX idx_milestones_peak ON sub_peak_milestones(sub_peak_id)');
    batch.execute(
        'CREATE INDEX idx_purchases_status ON purchases(status)');

    await batch.commit(noResult: true);
    _logger.i('数据库表结构创建完成 (v$version)');
  }

  /// 数据库迁移
  Future<void> _onUpgrade(Database db, int oldVersion, int newVersion) async {
    _logger.i('数据库迁移: v$oldVersion → v$newVersion');
    for (var v = oldVersion + 1; v <= newVersion; v++) {
      await _migrateTo(db, v);
    }
  }

  Future<void> _migrateTo(Database db, int version) async {
    switch (version) {
      case 2:
        // v1 → v2: 移除 anonymous_id 列（user_profile 改为单行表）
        // 由于 SQLite 不支持 DROP COLUMN 在旧版本，
        // 这里重建 user_profile 表
        await db.execute('DROP TABLE IF EXISTS user_profile_old');
        await db.execute('ALTER TABLE user_profile RENAME TO user_profile_old');
        await db.execute('''
          CREATE TABLE user_profile (
            id INTEGER PRIMARY KEY DEFAULT 1 CHECK(id = 1),
            traits_json TEXT NOT NULL DEFAULT '[]',
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
          )
        ''');
        // 迁移旧数据（如果存在）
        final oldRows = await db.query('user_profile_old', limit: 1);
        if (oldRows.isNotEmpty) {
          final old = oldRows.first;
          await db.insert('user_profile', {
            'id': 1,
            'traits_json': old['traits_json'] as String? ?? '[]',
            'created_at': old['created_at'] as int? ?? DateTime.now().millisecondsSinceEpoch,
            'updated_at': old['updated_at'] as int? ?? DateTime.now().millisecondsSinceEpoch,
          });
        }
        await db.execute('DROP TABLE IF EXISTS user_profile_old');
        _logger.i('迁移完成: v1 → v2 (user_profile 重构)');
        break;
    }
  }

  /// 关闭数据库连接
  Future<void> close() async {
    await _db?.close();
    _db = null;
    _logger.i('数据库已关闭');
  }
}
