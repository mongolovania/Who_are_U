import 'dart:convert';
import 'dart:io';

import 'package:encrypt/encrypt.dart' as enc;
import 'package:logger/logger.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

/// 加密文件存储管理器
///
/// 用于安全存储对话历史（完整 JSON/Markdown）。
/// 每个对话保存为独立的加密文件，支持按需解密读取。
///
/// 文件结构：
/// ```
/// conversations/
///   conv_abc123.enc      # 加密对话文件
///   conv_def456.enc
/// reports/
///   report_abc123.enc    # 加密决策报告
/// ```
class FileStore {
  final Logger _logger = Logger();

  late final Directory _conversationsDir;
  late final Directory _reportsDir;
  enc.Key? _encryptionKey;
  enc.IV? _iv;

  /// 初始化文件存储目录
  Future<void> initialize() async {
    final docsDir = await getApplicationDocumentsDirectory();

    _conversationsDir = Directory(p.join(docsDir.path, 'conversations'));
    if (!await _conversationsDir.exists()) {
      await _conversationsDir.create(recursive: true);
    }

    _reportsDir = Directory(p.join(docsDir.path, 'reports'));
    if (!await _reportsDir.exists()) {
      await _reportsDir.create(recursive: true);
    }

    _logger.i('文件存储初始化完成: ${docsDir.path}');
  }

  /// 设置加密密钥（从 SecureStore 获取后调用）
  void setEncryptionKey(String key) {
    // 从密钥派生出 256-bit AES 密钥和 IV
    final keyBytes = enc.Key.fromUtf8(key.padRight(32).substring(0, 32));
    _encryptionKey = keyBytes;
    _iv = enc.IV.fromUtf8(key.padRight(16).substring(0, 16));
  }

  // ─── 对话存储 ───────────────────────────────────

  /// 保存加密对话
  Future<File> saveConversation(String id, Map<String, dynamic> data) async {
    final json = jsonEncode(data);
    final encrypted = _encrypt(json);
    final file = File(p.join(_conversationsDir.path, 'conv_$id.enc'));
    await file.writeAsBytes(encrypted.bytes);
    _logger.d('对话已保存: $id');
    return file;
  }

  /// 读取并解密对话
  Future<Map<String, dynamic>> readConversation(String id) async {
    final file = File(p.join(_conversationsDir.path, 'conv_$id.enc'));
    if (!await file.exists()) {
      throw FileSystemException('对话文件不存在', file.path);
    }
    final bytes = await file.readAsBytes();
    final json = _decrypt(enc.Encrypted(bytes));
    return jsonDecode(json) as Map<String, dynamic>;
  }

  /// 删除对话文件
  Future<void> deleteConversation(String id) async {
    final file = File(p.join(_conversationsDir.path, 'conv_$id.enc'));
    if (await file.exists()) {
      await file.delete();
      _logger.d('对话已删除: $id');
    }
  }

  /// 列出所有对话 ID
  Future<List<String>> listConversations() async {
    final files = await _conversationsDir
        .list()
        .where((e) => e.path.endsWith('.enc'))
        .toList();
    return files
        .map((f) => p.basenameWithoutExtension(f.path).replaceFirst('conv_', ''))
        .toList()
      ..sort((a, b) => b.compareTo(a)); // 倒序
  }

  // ─── 报告存储 ───────────────────────────────────

  /// 保存加密决策报告
  Future<File> saveReport(String id, Map<String, dynamic> data) async {
    final json = jsonEncode(data);
    final encrypted = _encrypt(json);
    final file = File(p.join(_reportsDir.path, 'report_$id.enc'));
    await file.writeAsBytes(encrypted.bytes);
    return file;
  }

  /// 读取并解密决策报告
  Future<Map<String, dynamic>> readReport(String id) async {
    final file = File(p.join(_reportsDir.path, 'report_$id.enc'));
    if (!await file.exists()) {
      throw FileSystemException('报告文件不存在', file.path);
    }
    final bytes = await file.readAsBytes();
    final json = _decrypt(enc.Encrypted(bytes));
    return jsonDecode(json) as Map<String, dynamic>;
  }

  // ─── 导出 ───────────────────────────────────────

  /// 导出所有对话为未加密 JSON（用于备份）
  Future<String> exportAllUnencrypted() async {
    final all = <Map<String, dynamic>>[];
    final ids = await listConversations();
    for (final id in ids) {
      try {
        final conv = await readConversation(id);
        all.add(conv);
      } catch (e) {
        _logger.w('跳过损坏的对话: $id');
      }
    }
    return jsonEncode(all);
  }

  /// 获取存储大小（字节）
  Future<int> getTotalSize() async {
    var size = 0;
    for (final dir in [_conversationsDir, _reportsDir]) {
      await for (final entity in dir.list(recursive: true)) {
        if (entity is File) {
          size += await entity.length();
        }
      }
    }
    return size;
  }

  // ─── 加密/解密 ──────────────────────────────────

  enc.Encrypted _encrypt(String plaintext) {
    if (_encryptionKey == null) {
      throw StateError('加密密钥未设置，请先调用 setEncryptionKey()');
    }
    final encrypter = enc.Encrypter(enc.AES(_encryptionKey!));
    return encrypter.encrypt(plaintext, iv: _iv);
  }

  String _decrypt(enc.Encrypted encrypted) {
    if (_encryptionKey == null) {
      throw StateError('加密密钥未设置，请先调用 setEncryptionKey()');
    }
    final encrypter = enc.Encrypter(enc.AES(_encryptionKey!));
    return encrypter.decrypt(encrypted, iv: _iv);
  }
}
