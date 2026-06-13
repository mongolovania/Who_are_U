/// 6 Trait → 24原型映射
///
/// 从 Sprint 2 的 10题测试结果（3个PersonaTrait）映射到候选原型。
/// 每个 trait 映射到 2 个候选 Archetype ID。
///
/// 映射逻辑来自 WBS §7.1:
///   冒险家 → 探险家, 开拓者
///   守护者 → 守护者, 哨兵
///   点子王 → 发明家, 谋士
///   坚韧者 → 赴汤蹈火者, 隐者
///   共情者 → 治愈师, 倾听者
///   完美主义者 → 建筑师, 侦探
const Map<String, List<String>> traitToArchetypeIds = {
  'adventurer': ['explorer', 'pioneer'],
  'guardian': ['guardian', 'sentinel'],
  'idea_king': ['inventor', 'strategist'],
  'resilient': ['firewalker', 'hermit'],
  'empath': ['healer', 'listener'],
  'perfectionist': ['architect', 'detective'],
};

/// Trait key → 中文标签（用于调试/日志）
const Map<String, String> traitLabels = {
  'adventurer': '冒险家',
  'guardian': '守护者',
  'idea_king': '点子王',
  'resilient': '坚韧者',
  'empath': '共情者',
  'perfectionist': '完美主义者',
};
