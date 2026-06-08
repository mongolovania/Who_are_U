import 'package:bloc_test/bloc_test.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:who_are_u/features/onboarding/presentation/cubit/test_cubit.dart';
import 'package:who_are_u/features/onboarding/presentation/cubit/test_state.dart';

void main() {
  group('TestCubit', () {
    late TestCubit cubit;

    setUp(() {
      cubit = TestCubit();
    });

    tearDown(() {
      cubit.close();
    });

    test('初始状态: currentIndex=0, answers={} , status=answering', () {
      expect(cubit.state.currentIndex, equals(0));
      expect(cubit.state.answers, isEmpty);
      expect(cubit.state.status, equals(TestStatus.answering));
      expect(cubit.state.questions.length, equals(10));
    });

    blocTest<TestCubit, TestState>(
      'selectAnswer 更新当前题目的答案',
      build: () => TestCubit(),
      act: (cubit) => cubit.selectAnswer(4),
      expect: () => [
        predicate<TestState>((s) =>
            s.answers[0] == 4 &&
            s.currentIndex == 0),
      ],
    );

    blocTest<TestCubit, TestState>(
      'next 移动到下一题',
      build: () => TestCubit(),
      act: (cubit) {
        cubit.selectAnswer(3);
        cubit.next();
      },
      expect: () => [
        predicate<TestState>((s) => s.answers[0] == 3),
        predicate<TestState>((s) => s.currentIndex == 1),
      ],
    );

    blocTest<TestCubit, TestState>(
      'previous 返回上一题',
      build: () => TestCubit(),
      act: (cubit) {
        cubit.selectAnswer(3);
        cubit.next();
        cubit.previous();
      },
      expect: () => [
        isA<TestState>(),
        isA<TestState>(),
        predicate<TestState>((s) => s.currentIndex == 0),
      ],
    );

    blocTest<TestCubit, TestState>(
      '重复选同一题覆盖旧答案',
      build: () => TestCubit(),
      act: (cubit) {
        cubit.selectAnswer(2);
        cubit.selectAnswer(5);
      },
      expect: () => [
        predicate<TestState>((s) => s.answers[0] == 2),
        predicate<TestState>((s) => s.answers[0] == 5),
      ],
    );

    blocTest<TestCubit, TestState>(
      '第 10 题选完后 next 自动计算 → done',
      build: () => TestCubit(),
      act: (cubit) {
        // 回答所有 10 题
        for (var i = 0; i < 10; i++) {
          cubit.jumpTo(i);
          cubit.selectAnswer(4);
        }
        cubit.jumpTo(9); // 回到最后一题
        cubit.next(); // 触发计算
      },
      // jumpTo + selectAnswer 产生大量中间状态，仅验证最终状态
      verify: (cubit) {
        final state = cubit.state;
        expect(state.status, equals(TestStatus.done));
        expect(state.result, isNotNull);
        expect(state.result!.traits.length, equals(3));
      },
    );

    test('jumpTo 边界检查', () {
      cubit.jumpTo(5);
      expect(cubit.state.currentIndex, equals(5));

      cubit.jumpTo(-1); // 无效
      expect(cubit.state.currentIndex, equals(5));

      cubit.jumpTo(10); // 无效
      expect(cubit.state.currentIndex, equals(5));
    });

    test('first 不能 previous', () {
      cubit.previous();
      expect(cubit.state.currentIndex, equals(0));
    });
  });
}
