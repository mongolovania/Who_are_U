import 'dart:math';

import 'package:flutter/material.dart';

import '../../../../shared/models/user.dart';

/// 特质颜色映射
///
/// 每个特质有独立的暖色调，用于泡泡渐变填充。
const Map<String, Color> traitColors = {
  'adventurer': Color(0xFFE76F51),    // 暖橙
  'guardian': Color(0xFFC9A96E),      // 暖棕
  'idea_king': Color(0xFFF2CC8F),     // 暖金
  'resilient': Color(0xFFE07A5F),     // 暖铜
  'empath': Color(0xFFF4A261),        // 暖粉
  'perfectionist': Color(0xFFD4A5A5), // 暖紫
};

/// 特质泡泡组件
///
/// 圆形泡泡展示单个 [PersonaTrait]。
/// 大小根据 score 动态变化 (120-180px)。
/// 包含 emoji、label 和装饰性得分环形。
class TraitBubble extends StatefulWidget {
  final PersonaTrait trait;
  final int rank; // 0=最高分, 2=最低分

  const TraitBubble({
    super.key,
    required this.trait,
    required this.rank,
  });

  @override
  State<TraitBubble> createState() => _TraitBubbleState();
}

class _TraitBubbleState extends State<TraitBubble>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final Animation<double> _scaleAnim;
  late final Animation<double> _fadeAnim;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 600),
      vsync: this,
    );

    // 延迟弹出效果
    final delay = widget.rank * 300;
    final curvedAnim = CurvedAnimation(
      parent: _controller,
      curve: Interval(
        (delay / 1200).clamp(0.0, 0.9),
        1.0,
        curve: Curves.elasticOut,
      ),
    );

    _scaleAnim = Tween<double>(begin: 0.0, end: 1.0).animate(curvedAnim);
    _fadeAnim = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
        parent: _controller,
        curve: const Interval(0.0, 0.6, curve: Curves.easeIn),
      ),
    );

    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final color = traitColors[widget.trait.key] ?? const Color(0xFFE07A5F);

    // 大小根据 score 和 rank
    // rank 0 (最高分): ~170px, rank 1: ~150px, rank 2: ~130px
    final baseSize = 180.0 - widget.rank * 30;
    final size = max(120.0, baseSize * (0.7 + widget.trait.score * 0.3));

    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return Opacity(
          opacity: _fadeAnim.value,
          child: Transform.scale(
            scale: _scaleAnim.value,
            child: child,
          ),
        );
      },
      child: SizedBox(
        width: size,
        height: size,
        child: Stack(
          alignment: Alignment.center,
          children: [
            // 泡泡背景
            Container(
              width: size,
              height: size,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: RadialGradient(
                  center: const Alignment(-0.2, -0.2),
                  colors: [
                    color.withAlpha(230),
                    color.withAlpha(140),
                  ],
                ),
                boxShadow: [
                  BoxShadow(
                    color: color.withAlpha(60),
                    blurRadius: 24,
                    offset: const Offset(0, 8),
                  ),
                ],
              ),
            ),

            // 内容
            Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // emoji
                Text(
                  widget.trait.emoji,
                  style: TextStyle(
                    fontSize: size * 0.22,
                  ),
                ),
                const SizedBox(height: 4),
                // 标签
                Text(
                  widget.trait.label,
                  style: TextStyle(
                    fontSize: size * 0.13,
                    fontWeight: FontWeight.w700,
                    color: Colors.white,
                    shadows: [
                      Shadow(
                        color: Colors.black.withAlpha(40),
                        blurRadius: 4,
                      ),
                    ],
                  ),
                ),
              ],
            ),

            // 得分环形进度条
            SizedBox(
              width: size,
              height: size,
              child: CircularProgressIndicator(
                value: widget.trait.score,
                strokeWidth: 2.5,
                backgroundColor: Colors.white.withAlpha(30),
                valueColor: AlwaysStoppedAnimation<Color>(
                  Colors.white.withAlpha(180),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
