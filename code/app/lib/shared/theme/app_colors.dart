import 'package:flutter/material.dart';

/// 主题色彩常量
///
/// 品牌色以暖色调为主，传递温暖、陪伴感。
/// 山脉配色用于"我之山"可视化的 3 个区域。
class AppColors {
  AppColors._();

  // 品牌色
  static const Color primary = Color(0xFFE07A5F); // 暖橙
  static const Color primaryLight = Color(0xFFF2CC8F); // 暖杏
  static const Color secondary = Color(0xFF81B29A); // 森林绿
  static const Color accent = Color(0xFF3D405B); // 深沉蓝紫

  // 山脉区域
  static const Color mountainBase = Color(0xFF2D2D44); // 山脚迷雾
  static const Color mountainMid = Color(0xFFE07A5F); // 山腰暖光
  static const Color mountainPeak = Color(0xFFF2CC8F); // 山顶星空
  static const Color mountainFog = Color(0x80607080); // 迷雾层

  // 背景
  static const Color background = Color(0xFFFAF7F2); // 暖白
  static const Color surface = Color(0xFFFFFFFF);
  static const Color darkBackground = Color(0xFF1A1A2E);

  // 文字
  static const Color textPrimary = Color(0xFF2D2D44);
  static const Color textSecondary = Color(0xFF6B6B7B);
  static const Color textOnPrimary = Color(0xFFFFFFFF);

  // 语义色
  static const Color success = Color(0xFF81B29A);
  static const Color warning = Color(0xFFF2CC8F);
  static const Color error = Color(0xFFE07A5F);
  static const Color info = Color(0xFF3D405B);

  // 节点类型色
  static const Color nodeDecision = Color(0xFFE07A5F);
  static const Color nodeDifficulty = Color(0xFFF4A261);
  static const Color nodeSubpeak = Color(0xFF81B29A);
  static const Color nodeAchievement = Color(0xFFF2CC8F);
}
