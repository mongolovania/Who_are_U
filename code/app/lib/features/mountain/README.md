# 我之山可视化模块 (Mountain)

"我之山"成长山脉的可视化渲染。
每次深度对话、困难记录或副峰登顶都会在山脉上点亮新节点。

## 视觉分层

```
山顶星空 (Peak)    — 已探索区域，通透开阔
山腰暖光 (Mid)      — 正在探索，温暖引导
山脚迷雾 (Base)     — 未探索，保留神秘感
```

## 关键文件

- [mountain_painter.dart](presentation/painters/mountain_painter.dart) — 山体 CustomPainter
- [shadow_figure_painter.dart](presentation/painters/shadow_figure_painter.dart) — 独影剪影
- [node_painter.dart](presentation/painters/node_painter.dart) — 节点绘制
- [mountain_viewport.dart](presentation/widgets/mountain_viewport.dart) — 可缩放视口
- [fog_layer.dart](presentation/widgets/fog_layer.dart) — 迷雾效果
