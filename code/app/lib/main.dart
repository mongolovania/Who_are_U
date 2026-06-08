import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:logger/logger.dart';

import 'app.dart';
import 'core/config/app_config.dart';

import 'core/storage/database.dart';
import 'core/storage/secure_store.dart';

/// 应用入口 — 初始化核心基础设施后启动 App
///
/// 初始化顺序：
/// 1. 绑定 Flutter 引擎
/// 2. 初始化安全存储（Keychain / EncryptedSharedPreferences）
/// 3. 生成/获取加密密钥（256-bit CSPRNG）
/// 4. 初始化本地数据库（SQLite + SQLCipher 加密）
/// 5. 加载应用配置
/// 6. 注入全局依赖
/// 7. 启动 App 组件树
void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final logger = Logger(level: AppConfig.isDebugMode ? Level.debug : Level.info);

  try {
    // 1. 初始化安全存储
    final secureStore = SecureStore();
    await secureStore.initialize();

    // 2. 确保加密密钥已生成
    await secureStore.getEncryptionKey();

    // 3. 初始化加密数据库
    final database = AppDatabase(secureStore: secureStore);
    await database.initialize();

    // 4. 加载应用配置
    final appConfig = AppConfig.load();

    logger.i('App 初始化完成 — 环境: ${appConfig.environment}');

    // 5. 启动 App
    runApp(
      MultiRepositoryProvider(
        providers: [
          RepositoryProvider<SecureStore>.value(value: secureStore),
          RepositoryProvider<AppDatabase>.value(value: database),
          RepositoryProvider<AppConfig>.value(value: appConfig),
        ],
        child: const WhoAreUApp(),
      ),
    );
  } catch (error, stackTrace) {
    logger.e('App 初始化失败', error: error, stackTrace: stackTrace);
    runApp(const CriticalErrorApp());
  }
}
