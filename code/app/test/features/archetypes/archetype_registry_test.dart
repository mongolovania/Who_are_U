import 'package:flutter_test/flutter_test.dart';
import 'package:who_are_u/features/archetypes/data/archetype_registry.dart';
import 'package:who_are_u/features/archetypes/models/archetype.dart';
import 'package:who_are_u/shared/models/user.dart';

void main() {
  late ArchetypeRegistry registry;

  setUp(() {
    registry = ArchetypeRegistry();
  });

  group('ArchetypeRegistry', () {
    test('should have all 24 archetypes', () {
      expect(registry.all.length, 24);
    });

    test('should have 4 factions with 6 each', () {
      for (final faction in Faction.values) {
        final byFaction = registry.getByFaction(faction);
        expect(byFaction.length, 6, reason: '${faction.label} should have 6 archetypes');
      }
    });

    test('all archetype IDs should be unique', () {
      final ids = registry.all.map((a) => a.id).toList();
      expect(ids.toSet().length, ids.length);
    });

    group('fromTraits', () {
      test('adventurer → explorer + pioneer', () {
        final traits = [
          PersonaTrait(key: 'adventurer', label: '冒险家', emoji: '🧗', description: '', score: 0.8),
        ];
        final result = registry.fromTraits(traits);
        expect(result.length, greaterThan(0));
        expect(result.any((a) => a.id == 'explorer' || a.id == 'pioneer'), isTrue);
      });

      test('guardian → guardian + sentinel', () {
        final traits = [
          PersonaTrait(key: 'guardian', label: '守护者', emoji: '🛡️', description: '', score: 0.8),
        ];
        final result = registry.fromTraits(traits);
        expect(result.any((a) => a.id == 'guardian' || a.id == 'sentinel'), isTrue);
      });

      test('idea_king → inventor + strategist', () {
        final traits = [
          PersonaTrait(key: 'idea_king', label: '点子王', emoji: '💡', description: '', score: 0.8),
        ];
        final result = registry.fromTraits(traits);
        expect(result.any((a) => a.id == 'inventor' || a.id == 'strategist'), isTrue);
      });

      test('resilient → firewalker + hermit', () {
        final traits = [
          PersonaTrait(key: 'resilient', label: '坚韧者', emoji: '🪨', description: '', score: 0.8),
        ];
        final result = registry.fromTraits(traits);
        expect(result.any((a) => a.id == 'firewalker' || a.id == 'hermit'), isTrue);
      });

      test('empath → healer + listener', () {
        final traits = [
          PersonaTrait(key: 'empath', label: '共情者', emoji: '💛', description: '', score: 0.8),
        ];
        final result = registry.fromTraits(traits);
        expect(result.any((a) => a.id == 'healer' || a.id == 'listener'), isTrue);
      });

      test('perfectionist → architect + detective', () {
        final traits = [
          PersonaTrait(key: 'perfectionist', label: '完美主义者', emoji: '✨', description: '', score: 0.8),
        ];
        final result = registry.fromTraits(traits);
        expect(result.any((a) => a.id == 'architect' || a.id == 'detective'), isTrue);
      });

      test('multiple traits → up to 3 archetypes, no duplicates', () {
        final traits = [
          PersonaTrait(key: 'empath', label: '共情者', emoji: '💛', description: '', score: 0.9),
          PersonaTrait(key: 'adventurer', label: '冒险家', emoji: '🧗', description: '', score: 0.7),
          PersonaTrait(key: 'perfectionist', label: '完美主义者', emoji: '✨', description: '', score: 0.5),
        ];
        final result = registry.fromTraits(traits);
        expect(result.length, lessThanOrEqualTo(3));
        // No duplicates
        final ids = result.map((a) => a.id).toSet();
        expect(ids.length, result.length);
      });

      test('empty traits → empty result', () {
        final result = registry.fromTraits([]);
        expect(result, isEmpty);
      });
    });
  });
}
