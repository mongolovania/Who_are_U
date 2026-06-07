import '../models/test_question.dart';

/// 10 题初始测试题库
///
/// 每题是 5 点 Likert 量表陈述句。
/// 用户选择同意程度：1=非常不同意 → 5=非常同意。
///
/// 3 个维度各 3 道题 + 1 道混合题 (Q10 计入所有维度)：
/// - 行动倾向 (Q1, Q4, Q7): 冒险家 ↔ 守护者
/// - 思维模式 (Q2, Q5, Q8): 点子王 ↔ 坚韧者
/// - 情感倾向 (Q3, Q6, Q9): 共情者 ↔ 完美主义者
const List<TestQuestion> testQuestions = [
  TestQuestion(
    id: 'q1',
    index: 0,
    text: '面对陌生的环境，我会兴奋地探索每一个角落',
    dimension: PersonalityDimension.action,
  ),
  TestQuestion(
    id: 'q2',
    index: 1,
    text: '遇到问题时，我喜欢先头脑风暴出各种可能性',
    dimension: PersonalityDimension.thinking,
  ),
  TestQuestion(
    id: 'q3',
    index: 2,
    text: '做重要决定时，我首先考虑的是对他人的影响',
    dimension: PersonalityDimension.emotion,
  ),
  TestQuestion(
    id: 'q4',
    index: 3,
    text: '相比稳定的生活，我更渴望充满未知的冒险',
    dimension: PersonalityDimension.action,
  ),
  TestQuestion(
    id: 'q5',
    index: 4,
    text: '我经常同时进行多个项目或想法，而不是一次只专注一件事',
    dimension: PersonalityDimension.thinking,
  ),
  TestQuestion(
    id: 'q6',
    index: 5,
    text: '当我感到压力时，我更愿意找人倾诉而不是独自消化',
    dimension: PersonalityDimension.emotion,
  ),
  TestQuestion(
    id: 'q7',
    index: 6,
    text: '面对"刺激但不确定"和"稳定但可预测"，我更倾向前者',
    dimension: PersonalityDimension.action,
  ),
  TestQuestion(
    id: 'q8',
    index: 7,
    text: '做计划时，我倾向于先列出所有可能性再逐一筛选',
    dimension: PersonalityDimension.thinking,
  ),
  TestQuestion(
    id: 'q9',
    index: 8,
    text: '看到朋友做了我不认同的选择，我会先试着理解他的感受',
    dimension: PersonalityDimension.emotion,
  ),
  TestQuestion(
    id: 'q10',
    index: 9,
    text: '回顾过去，我最大的驱动力来自好奇心而非责任感',
    dimension: PersonalityDimension.action,
  ),
];

/// 所有维度列表（Q10 计入所有维度）
const allDimensions = [
  PersonalityDimension.action,
  PersonalityDimension.thinking,
  PersonalityDimension.emotion,
];
