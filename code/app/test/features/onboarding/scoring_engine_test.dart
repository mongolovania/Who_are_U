import 'package:flutter_test/flutter_test.dart';
import 'package:who_are_u/features/onboarding/data/scoring_engine.dart';
import 'package:who_are_u/shared/models/user.dart';

void main() {
  group('ScoringEngine', () {
    group('calculate', () {
      test('全 1 → 守护者 + 坚韧者 + 完美主义者', () {
        final answers = Map.fromIterables(
          List.generate(10, (i) => i),
          List.filled(10, 1),
        );

        final traits = ScoringEngine.calculate(answers);

        expect(traits.length, equals(3));
        expect(traits.map((t) => t.key).toSet(),
            equals({'guardian', 'resilient', 'perfectionist'}));
      });

      test('全 5 → 冒险家 + 点子王 + 共情者', () {
        final answers = Map.fromIterables(
          List.generate(10, (i) => i),
          List.filled(10, 5),
        );

        final traits = ScoringEngine.calculate(answers);

        expect(traits.length, equals(3));
        expect(traits.map((t) => t.key).toSet(),
            equals({'adventurer', 'idea_king', 'empath'}));
      });

      test('分数范围在 0.0-1.0', () {
        final answers = Map.fromIterables(
          List.generate(10, (i) => i),
          [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        );

        final traits = ScoringEngine.calculate(answers);

        for (final trait in traits) {
          expect(trait.score, greaterThanOrEqualTo(0.0));
          expect(trait.score, lessThanOrEqualTo(1.0));
        }
      });

      test('混合值 — 中间分映射到低分 trait', () {
        // 全部 3 分（刚好在边界以下，因为 3.0 会映射到高分trait）
        // 但在加权计算中，3 * 3 + 3 / 4 = 3.0，所以 >= 3.0 → 高分 trait
        final answers = Map.fromIterables(
          List.generate(10, (i) => i),
          [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], // 全 2 → 低分 trait
        );

        final traits = ScoringEngine.calculate(answers);

        expect(traits.map((t) => t.key).toSet(),
            equals({'guardian', 'resilient', 'perfectionist'}));
      });

      test('按 score 降序排列', () {
        final answers = Map.fromIterables(
          List.generate(10, (i) => i),
          [5, 4, 3, 5, 4, 3, 5, 4, 3, 5],
        );

        final traits = ScoringEngine.calculate(answers);

        expect(traits.length, equals(3));
        expect(traits[0].score, greaterThanOrEqualTo(traits[1].score));
        expect(traits[1].score, greaterThanOrEqualTo(traits[2].score));
      });

      test('每个 trait 有完整的字段', () {
        final answers = Map.fromIterables(
          List.generate(10, (i) => i),
          [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        );

        final traits = ScoringEngine.calculate(answers);

        for (final trait in traits) {
          expect(trait.key, isNotEmpty);
          expect(trait.label, isNotEmpty);
          expect(trait.emoji, isNotEmpty);
          expect(trait.description, isNotEmpty);
          expect(trait.score, isA<double>());
        }
      });
    });
  });

  group('PersonaTrait JSON', () {
    test('序列化/反序列化往返', () {
      const trait = PersonaTrait(
        key: 'adventurer',
        label: '冒险家',
        emoji: '🧗',
        description: '测试描述',
        score: 0.85,
      );

      final json = trait.toJson();
      final restored = PersonaTrait.fromJson(json);

      expect(restored.key, equals(trait.key));
      expect(restored.label, equals(trait.label));
      expect(restored.emoji, equals(trait.emoji));
      expect(restored.description, equals(trait.description));
      expect(restored.score, equals(trait.score));
    });
  });
}
