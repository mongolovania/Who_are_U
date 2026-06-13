import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:who_are_u/features/archetypes/data/archetypes.dart';
import 'package:who_are_u/features/masters/data/master_registry.dart';
import 'package:who_are_u/features/masters/data/selection_engine.dart';
import 'package:who_are_u/features/masters/models/master.dart';

void main() {
  late MasterRegistry registry;
  late SelectionEngine engine;

  setUpAll(() async {
    // Load masters from file system (rootBundle not available in unit tests)
    final file = File('assets/masters.json');
    final jsonStr = await file.readAsString();

    registry = MasterRegistry();
    registry.loadFromString(jsonStr);

    engine = SelectionEngine(registry);
    engine.buildMatrix();
  });

  group('Master data', () {
    test('should load 50 masters', () {
      expect(registry.all.length, 50);
    });

    test('should have 4 domains with masters', () {
      for (final domain in Domain.values) {
        final masters = registry.getByDomain(domain);
        expect(masters.length, greaterThan(0),
            reason: '${domain.label} should have masters');
      }
    });

    test('all master IDs should be unique', () {
      final ids = registry.all.map((m) => m.id).toList();
      expect(ids.toSet().length, ids.length);
    });

    test('can find master by ID', () {
      final munger = registry.getById('munger');
      expect(munger, isNotNull);
      expect(munger!.nameCn, '查理·芒格');
    });
  });

  group('SelectionEngine', () {
    test('select with archetypes returns 7 masters', () {
      final explorer = allArchetypes.firstWhere((a) => a.id == 'explorer');
      final result = engine.select(archetypes: [explorer]);
      expect(result.length, 7);
    });

    test('select with domain filter returns domain-relevant masters', () {
      final strategist = allArchetypes.firstWhere((a) => a.id == 'strategist');
      final result = engine.select(
        archetypes: [strategist],
        domain: Domain.economy,
      );
      expect(result.length, 7);
      final economyCount = result.where((m) => m.domain == Domain.economy).length;
      expect(economyCount, greaterThan(0));
    });

    test('select with multiple archetypes returns no duplicates', () {
      final archetypes = [
        allArchetypes.firstWhere((a) => a.id == 'healer'),
        allArchetypes.firstWhere((a) => a.id == 'explorer'),
      ];
      final result = engine.select(archetypes: archetypes);
      final ids = result.map((m) => m.id).toSet();
      expect(ids.length, 7);
    });

    test('user mentioned master is forced into top 7', () {
      final pioneer = allArchetypes.firstWhere((a) => a.id == 'pioneer');
      final result = engine.select(
        archetypes: [pioneer],
        userMentions: ['查理·芒格'],
      );
      expect(result.any((m) => m.nameCn == '查理·芒格'), isTrue);
    });

    test('select with empty archetypes returns 7 random masters', () {
      final result = engine.select(archetypes: []);
      expect(result.length, 7);
    });

    test('all results have valid data', () {
      final sage = allArchetypes.firstWhere((a) => a.id == 'sage');
      final result = engine.select(archetypes: [sage]);
      for (final master in result) {
        expect(master.nameCn, isNotEmpty);
        expect(master.methodology, isNotEmpty);
        expect(master.goldenQuote, isNotEmpty);
      }
    });
  });
}
