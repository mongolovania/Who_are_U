import 'dart:convert';
import 'dart:math';

import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:logger/logger.dart';

/// 安全密钥存储管理器
///
/// 使用平台级安全存储：
/// - iOS: Keychain Services
/// - Android: EncryptedSharedPreferences
///
/// 管理的密钥：
/// - 加密密钥 (encryption_key): 供 AppDatabase + FileStore 使用
/// - 备份加密密钥 (backup_encryption_key): 供数据导出加密使用
class SecureStore {
  static const _dbEncryptionKey = 'encryption_key';
  static const _backupKey = 'backup_encryption_key';

  late final FlutterSecureStorage _storage;
  final Logger _logger = Logger();

  /// 初始化安全存储
  Future<void> initialize() async {
    _storage = const FlutterSecureStorage(
      aOptions: AndroidOptions(
        encryptedSharedPreferences: true,
      ),
      iOptions: IOSOptions(
        accessibility: KeychainAccessibility.first_unlock_this_device,
      ),
    );
    _logger.i('安全存储初始化完成');
  }

  // ─── 加密密钥 ──────────────────────────────────

  /// 获取统一的加密密钥（供 AppDatabase + FileStore 使用）
  ///
  /// 首次调用时使用密码学安全随机数生成 256-bit 密钥，
  /// 存储在 Keychain/EncryptedSharedPreferences 中。
  /// 后续调用返回相同密钥。
  Future<String> getEncryptionKey() async {
    return _getOrCreate(_dbEncryptionKey, 'encryption_key');
  }

  // ─── 备份加密密钥 ────────────────────────────────

  /// 获取或生成备份加密密钥
  Future<String> getOrCreateBackupKey() async {
    return _getOrCreate(_backupKey, 'backup_key');
  }

  // ─── 通用 ───────────────────────────────────────

  /// 获取或创建密钥
  ///
  /// 使用 [Random.secure()] 生成密码学安全的 256-bit 随机密钥，
  /// Base64URL 编码后存储。
  Future<String> _getOrCreate(String key, String label) async {
    final existing = await _storage.read(key: key);
    if (existing != null && existing.isNotEmpty) {
      _logger.d('使用已有密钥: $label');
      return existing;
    }

    // 使用密码学安全的随机数生成器生成 256-bit 密钥
    final random = Random.secure();
    final bytes = List<int>.generate(32, (_) => random.nextInt(256));
    final generated = base64Url.encode(bytes);

    await _storage.write(key: key, value: generated);
    _logger.i('生成新密钥: $label');
    return generated;
  }

  /// 存储任意密钥
  Future<void> write(String key, String value) async {
    await _storage.write(key: key, value: value);
  }

  /// 读取任意密钥
  Future<String?> read(String key) async {
    return _storage.read(key: key);
  }

  /// 删除密钥
  Future<void> delete(String key) async {
    await _storage.delete(key: key);
  }

  /// 清除所有数据（用户删除账户时调用）
  Future<void> clearAll() async {
    await _storage.deleteAll();
    _logger.i('所有安全存储数据已清除');
  }
}
