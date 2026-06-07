/// 主题微调配体
///
/// 区别于 app_theme.dart 的 Material ThemeData 定义，
/// 此文件管理运行时主题偏好和自定义配置。

/// 山脉区域渐变色定义
class MountainGradient {
  const MountainGradient._();

  static const List<int> baseColors = [
    0xFF1A1A2E, // 最深处
    0xFF2D2D44, // 山脚
    0xFF3D405B, // 过渡
  ];

  static const List<int> midColors = [
    0xFFE07A5F, // 暖橙
    0xFFF4A261, // 暖黄
    0xFFF2CC8F, // 暖杏
  ];

  static const List<int> peakColors = [
    0xFFF2CC8F, // 暖杏
    0xFFFFF3E0, // 星空底色
    0xFFE8F5E9, // 通透绿
  ];
}
