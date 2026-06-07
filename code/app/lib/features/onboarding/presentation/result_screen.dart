import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../../../shared/theme/app_colors.dart';
import '../../models/test_result.dart';
import 'widgets/trait_bubble.dart';

/// 特质泡泡结果展示
///
/// 展示 3 个泡泡，从大到小排列（score 最高的最大）。
/// 用户看到自己是什么样的人，然后进入我之山。
class TestResultScreen extends StatelessWidget {
  final TestResult result;

  const TestResultScreen({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final traits = result.traits; // 已按 score 降序排列

    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              AppColors.background,
              Color(0xFFF0E8D8), // 暖杏底色调
              Color(0xFFEDE0D0),
            ],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              const Spacer(flex: 2),

              // 顶部标题
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 32),
                child: Text(
                  '你身上有三种\n珍贵的力量 ✨',
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.headlineLarge?.copyWith(
                        height: 1.3,
                      ),
                ),
              ),
              const SizedBox(height: 8),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 32),
                child: Text(
                  '这不是标签，而是你此刻的底色。\n它们会在你的山中生根、成长。',
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        height: 1.5,
                      ),
                ),
              ),

              const SizedBox(height: 36),

              // 泡泡展示
              SizedBox(
                height: 220,
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: List.generate(
                    traits.length,
                    (i) => Padding(
                      padding: EdgeInsets.only(
                        top: i == 0 ? 0 : 24 + i * 12.0,
                        left: i > 0 ? 8 : 0,
                        right: i < traits.length - 1 ? 8 : 0,
                      ),
                      child: TraitBubble(
                        trait: traits[i],
                        rank: i,
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 20),

              // 特质描述卡片
              _TraitDescriptionCard(traits: traits),

              const Spacer(),

              // 进入我之山按钮
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 32),
                child: SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: () {
                      // 保存画像后跳转到山脉
                      // TODO Sprint 5: 调用 PersonaRepository.save
                      context.go('/mountain');
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppColors.primary,
                      foregroundColor: AppColors.textOnPrimary,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                      padding: const EdgeInsets.symmetric(vertical: 18),
                    ),
                    child: const Text(
                      '进入我的山 →',
                      style: TextStyle(fontSize: 18),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 40),
            ],
          ),
        ),
      ),
    );
  }
}

/// 特质描述卡片
///
/// 展示最高分特质的详细描述。
class _TraitDescriptionCard extends StatelessWidget {
  final List<PersonaTrait> traits;

  const _TraitDescriptionCard({required this.traits});

  @override
  Widget build(BuildContext context) {
    // 展示最高分特质
    final top = traits.first;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 40),
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: AppColors.surface.withAlpha(180),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  top.emoji,
                  style: const TextStyle(fontSize: 24),
                ),
                const SizedBox(width: 8),
                Text(
                  '你最突出的特质是「${top.label}」',
                  style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                        fontWeight: FontWeight.w600,
                      ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              top.description,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    height: 1.6,
                  ),
            ),
          ],
        ),
      ),
    );
  }
}
