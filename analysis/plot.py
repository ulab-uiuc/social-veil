#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# 数据
barrier_types = ['Semantic', 'Cultural', 'Emotional']
accuracies = [0.65, 0.63, 0.67]
ci_lower = [0.54, 0.53, 0.54]
ci_upper = [0.76, 0.73, 0.78]

# 计算误差条
errors_lower = [acc - ci_lower[i] for i, acc in enumerate(accuracies)]
errors_upper = [ci_upper[i] - acc for i, acc in enumerate(accuracies)]
errors = [errors_lower, errors_upper]

# 创建图表，显著缩小尺寸
fig, ax = plt.subplots(figsize=(3.5, 3))  # 从(5,5)缩小到(3.5,3)

# 使用更有质感的配色方案
colors = ['#FF7B7B', '#5DADE2', '#85E085']
shadow_colors = ['#E85555', '#3498DB', '#5CB85C']

# 先创建阴影效果（稍微偏移的较深色柱子）
shadow_bars = ax.bar([i + 0.002 for i in range(len(barrier_types))],
                     [acc - 0.002 for acc in accuracies],
                     color=shadow_colors,
                     alpha=0.3,
                     width=0.5,  # 从0.7缩小到0.5
                     zorder=1)

# 创建主柱状图，缩小柱子宽度
bars = ax.bar(barrier_types, accuracies,
              color=colors,
              alpha=0.9,
              edgecolor='white',
              linewidth=1.2,  # 稍微减小边框线宽
              width=0.5,      # 从0.7缩小到0.5
              zorder=2)

# 为每个柱子添加渐变效果
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

for i, (bar, color) in enumerate(zip(bars, colors)):
    x, y = bar.get_xy()
    w, h = bar.get_width(), bar.get_height()
    
    gradient = ax.imshow([[1, 0.7]], extent=[x, x+w, y, y+h],
                        aspect='auto', cmap=LinearSegmentedColormap.from_list('', ['white', color]),
                        alpha=0.3, zorder=1.5)

# 添加误差条，使用更细的样式
ax.errorbar(range(len(barrier_types)), accuracies,
            yerr=errors,
            fmt='none',
            color='black',
            capsize=2.5,    # 从3减小到2.5
            capthick=1.0,   # 从1.2减小到1.0
            elinewidth=1.0, # 从1.2减小到1.0
            zorder=3)

# 设置标题和标签，更紧凑的间距和更小字体
ax.set_title('Human Evaluation by Barrier Type',
             fontsize=10, fontweight='bold', pad=8)  # 从13减小到10
ax.set_ylabel('Accuracy', fontsize=9, fontweight='bold')  # 从11减小到9
ax.set_xlabel('Barrier Type', fontsize=9, fontweight='bold')  # 从11减小到9

# 调整Y轴范围和刻度
ax.set_ylim(0, 0.9)
ax.set_yticks(np.arange(0, 1.0, 0.1))

# 添加网格线
ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# 美化边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# 调整刻度标签，更小字体
ax.tick_params(axis='both', which='major', labelsize=8)  # 从10减小到8
ax.tick_params(axis='x', pad=2)
ax.tick_params(axis='y', pad=2)

# 设置x轴的范围，适应更窄的柱子
ax.set_xlim(-0.5, 2.5)  # 从(-0.6, 2.6)调整到(-0.5, 2.5)

# 在柱子上方添加数值标签，更小字体
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.015,
            f'{acc:.2f}',
            ha='center', va='bottom',
            fontweight='bold',
            fontsize=8)  # 从10减小到8

# 调整布局，更紧凑
plt.tight_layout(pad=0.8)  # 从1.0减小到0.8

# 保存图片
plt.savefig('human_evaluation_by_barrier_type_compact.png',
            dpi=300, bbox_inches='tight', facecolor='white')