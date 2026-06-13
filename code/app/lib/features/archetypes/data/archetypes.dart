import '../models/archetype.dart';

/// 24原型完整定义
///
/// 4阵营 × 6原型 = 24型
/// 数据来源：总体技术方案 §7.1
const List<Archetype> allArchetypes = [
  // ═══════════════════════════════════════════════════════
  // 🌿 滋养者 Nurturers (1-6)
  // ═══════════════════════════════════════════════════════
  Archetype(
    id: 'gardener',
    nameCn: '花匠',
    nameEn: 'Gardener',
    emoji: '🌸',
    faction: Faction.nurturer,
    drive: '培育与呵护',
    oneLiner: '你浇灌什么，什么就会生长',
  ),
  Archetype(
    id: 'healer',
    nameCn: '治愈师',
    nameEn: 'Healer',
    emoji: '💚',
    faction: Faction.nurturer,
    drive: '修复与和解',
    oneLiner: '破碎之处，光会照进来',
  ),
  Archetype(
    id: 'listener',
    nameCn: '倾听者',
    nameEn: 'Listener',
    emoji: '👂',
    faction: Faction.nurturer,
    drive: '共情与理解',
    oneLiner: '你说的每一句，我都真的在听',
  ),
  Archetype(
    id: 'guardian',
    nameCn: '守护者',
    nameEn: 'Guardian',
    emoji: '🛡',
    faction: Faction.nurturer,
    drive: '捍卫与奉献',
    oneLiner: '你守护的人，也守护着你',
  ),
  Archetype(
    id: 'elf',
    nameCn: '精灵',
    nameEn: 'Elf',
    emoji: '🧝',
    faction: Faction.nurturer,
    drive: '灵动与纯真',
    oneLiner: '世界在认真的人眼里是魔法',
  ),
  Archetype(
    id: 'poet',
    nameCn: '诗人',
    nameEn: 'Poet',
    emoji: '🎭',
    faction: Faction.nurturer,
    drive: '表达与美',
    oneLiner: '你把感受变成别人能触摸的东西',
  ),

  // ═══════════════════════════════════════════════════════
  // ⚔️ 行动者 Warriors (7-12)
  // ═══════════════════════════════════════════════════════
  Archetype(
    id: 'hero',
    nameCn: '英雄',
    nameEn: 'Hero',
    emoji: '🦸',
    faction: Faction.warrior,
    drive: '勇气与正义',
    oneLiner: '不是因为不怕，而是怕了也要上',
  ),
  Archetype(
    id: 'firewalker',
    nameCn: '赴汤蹈火者',
    nameEn: 'Firewalker',
    emoji: '🔥',
    faction: Faction.warrior,
    drive: '冒险与牺牲',
    oneLiner: '趟过火焰的人，脚下有光',
  ),
  Archetype(
    id: 'pioneer',
    nameCn: '开拓者',
    nameEn: 'Pioneer',
    emoji: '🚀',
    faction: Faction.warrior,
    drive: '先行与引领',
    oneLiner: '别人看到荒野，你看到路',
  ),
  Archetype(
    id: 'revolutionary',
    nameCn: '革命者',
    nameEn: 'Revolutionary',
    emoji: '⚡',
    faction: Faction.warrior,
    drive: '颠覆与重建',
    oneLiner: '旧的碎了，不等于世界完了',
  ),
  Archetype(
    id: 'avenger',
    nameCn: '复仇者',
    nameEn: 'Avenger',
    emoji: '⚔',
    faction: Faction.warrior,
    drive: '执念与裁决',
    oneLiner: '你说的"算了"，其实从来没有',
  ),
  Archetype(
    id: 'explorer',
    nameCn: '探险家',
    nameEn: 'Explorer',
    emoji: '🧭',
    faction: Faction.warrior,
    drive: '未知与发现',
    oneLiner: '地图的边界之外，才是你要去的地方',
  ),

  // ═══════════════════════════════════════════════════════
  // 🔮 智者 Sages (13-18)
  // ═══════════════════════════════════════════════════════
  Archetype(
    id: 'sage',
    nameCn: '贤者',
    nameEn: 'Sage',
    emoji: '📜',
    faction: Faction.sage,
    drive: '智慧与沉淀',
    oneLiner: '你不需要所有答案，只需要一个好问题',
  ),
  Archetype(
    id: 'strategist',
    nameCn: '谋士',
    nameEn: 'Strategist',
    emoji: '♟',
    faction: Faction.sage,
    drive: '布局与耐心',
    oneLiner: '每一步都算数，包括退后的那一步',
  ),
  Archetype(
    id: 'prophet',
    nameCn: '预言家',
    nameEn: 'Prophet',
    emoji: '🔮',
    faction: Faction.sage,
    drive: '远见与直觉',
    oneLiner: '别人看现在，你已经看到了以后',
  ),
  Archetype(
    id: 'detective',
    nameCn: '侦探',
    nameEn: 'Detective',
    emoji: '🔍',
    faction: Faction.sage,
    drive: '真相与解谜',
    oneLiner: '表面之下，总有另一个故事',
  ),
  Archetype(
    id: 'inventor',
    nameCn: '发明家',
    nameEn: 'Inventor',
    emoji: '⚙',
    faction: Faction.sage,
    drive: '创新与实用',
    oneLiner: '一个问题就是一个还没做出来的东西',
  ),
  Archetype(
    id: 'hermit',
    nameCn: '隐者',
    nameEn: 'Hermit',
    emoji: '🏔',
    faction: Faction.sage,
    drive: '内省与独立',
    oneLiner: '独处不是逃避，是回去找自己',
  ),

  // ═══════════════════════════════════════════════════════
  // 🎨 创变者 Creators (19-24)
  // ═══════════════════════════════════════════════════════
  Archetype(
    id: 'alchemist',
    nameCn: '炼金术士',
    nameEn: 'Alchemist',
    emoji: '⚗',
    faction: Faction.creator,
    drive: '转化与实验',
    oneLiner: '你触碰的东西都会变成别的东西',
  ),
  Archetype(
    id: 'magician',
    nameCn: '魔术师',
    nameEn: 'Magician',
    emoji: '🎪',
    faction: Faction.creator,
    drive: '惊喜与创造',
    oneLiner: '在你手里，普通的事变成魔法',
  ),
  Archetype(
    id: 'architect',
    nameCn: '建筑师',
    nameEn: 'Architect',
    emoji: '🏛',
    faction: Faction.creator,
    drive: '构建与秩序',
    oneLiner: '从第一块砖开始，一座城',
  ),
  Archetype(
    id: 'dreamer',
    nameCn: '梦想家',
    nameEn: 'Dreamer',
    emoji: '🌙',
    faction: Faction.creator,
    drive: '想象与可能',
    oneLiner: '你先梦到，然后才做到',
  ),
  Archetype(
    id: 'pyromancer',
    nameCn: '驭火者',
    nameEn: 'Pyromancer',
    emoji: '🌋',
    faction: Faction.creator,
    drive: '控制与力量',
    oneLiner: '不是不怕火，是学会了和火共处',
  ),
  Archetype(
    id: 'sentinel',
    nameCn: '哨兵',
    nameEn: 'Sentinel',
    emoji: '👁',
    faction: Faction.creator,
    drive: '警觉与边界',
    oneLiner: '你的沉默不是没有话说',
  ),
];
