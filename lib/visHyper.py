import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 加载 CSV 数据
# datax = [1,2,3,4,5,4,3,2,4,5,1,2]
# datay = [3,4,5,4,6,7,8,9,2,4,5,6]
#
# # 定义颜色
# orange = [0.91, 0.41, 0.17]
# blue = [0, 0.4470, 0.7410]
#
# # 创建图形
# fig, ax = plt.subplots(figsize=(8, 3))
# ax.set_xlabel('Month')
#
# # 左边 Y 轴（Data amount）
# ax = ax.twinx()  # 创建一个共享 x 轴的右边 Y 轴
# ax.plot(datax, '-', color=blue, linewidth=2, label='rmse')
# ax.set_ylabel('Data amount (× 10^6)')
# ax.set_ylim([0, 6])
# ax.set_yticks(np.arange(0, 6, 1))
# # ax1.set_yticklabels([f'{i/1e6:.1f}' for i in np.arange(0, 1.6e+6 + 0.4e+6, 0.4e+6)])
#
# # 右边 Y 轴（Completeness）
# ax1 = ax.twinx()  # 使用共享的 x 轴
# ax1.spines['right'].set_position(('outward', 60))  # 为了让右边的 Y 轴和左边的稍微分开
# ax1.plot(datay, '-', color=orange, linewidth=2, label='lb')
# ax1.set_ylabel('Completeness')
# ax1.set_ylim([0, 10])
# ax1.set_yticks(np.arange(0, 10, 1))
# # ax2.set_yticklabels([f'{i:.2f}' for i in np.arange(0, 1.25, 0.25)])
#
# # 设置 X 轴标签
# ax.set_xticks(np.arange(0, 12, 1))
# ax.set_xticklabels(['Apr.', 'Aug.', 'Dec.', 'Apr.', 'Aug.', 'Dec.', 'Apr.', 'Aug.', 'Dec.', 'Apr.', 'Aug.', 'Dec.'])
#
# # 格式化网格
# ax.grid(True, linestyle='-.')
# # ax1.grid(True, linestyle='-.')
# # ax2.grid(True, linestyle='-.')
# ax.tick_params(axis='y', labelsize=12, labelcolor=blue)  # 设置左侧 Y 轴刻度标签颜色为蓝色
# # ax1.tick_params(axis='y', labelsize=12, labelcolor=blue)  # 设置左侧 Y 轴刻度标签颜色为蓝色
# ax1.tick_params(axis='y', labelsize=12, labelcolor=orange)
#
# # 保存为 PDF 文件
# # plt.tight_layout()
# # plt.savefig('nyc_data_completeness.pdf', format='pdf')
# plt.show()

# 模拟数据
x = np.arange(1, 13)  # X 轴：1 到 12 (假设为月份)
y1 = np.random.randint(1, 10, 12)  # Y 轴 1：数据量
y2 = np.random.randint(100, 200, 12)  # Y 轴 2：另外一组数据

# 创建图形
fig, ax1 = plt.subplots(figsize=(8, 6))

# 左侧 Y 轴：绘制第一条折线图
ax1.plot(x, y1, color='tab:blue', label='Data 1', linewidth=2)
ax1.set_xlabel('Month', fontsize=14)  # 设置 X 轴标签
ax1.set_ylabel('Data 1', color='tab:blue', fontsize=14)  # 设置左侧 Y 轴标签
ax1.tick_params(axis='y', labelcolor='tab:blue')  # 设置左侧 Y 轴刻度标签颜色

# 设置左侧 Y 轴的显示范围和刻度
ax1.set_ylim([0, 10])  # 设置左侧 Y 轴的显示范围为 [0, 10]
ax1.set_yticks(np.arange(0, 11, 1))  # 设置左侧 Y 轴刻度：从 0 到 10，步长为 1

# 创建右侧 Y 轴（共享 X 轴）
ax2 = ax1.twinx()  # 创建一个共享 X 轴的右侧 Y 轴
ax2.plot(x, y2, color='tab:orange', label='Data 2', linewidth=2)
ax2.set_ylabel('Data 2', color='tab:orange', fontsize=14)  # 设置右侧 Y 轴标签
ax2.tick_params(axis='y', labelcolor='tab:orange')  # 设置右侧 Y 轴刻度标签颜色

# 设置右侧 Y 轴的显示范围和刻度
ax2.set_ylim([100, 200])  # 设置右侧 Y 轴的显示范围为 [100, 200]
ax2.set_yticks(np.arange(100, 201, 20))  # 设置右侧 Y 轴刻度：从 100 到 200，步长为 20

# 设置 X 轴刻度
ax1.set_xticks(x)  # 设置 X 轴刻度位置
ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)

# 格式化网格
ax1.grid(True, linestyle='-.', alpha=0.6)
fig.tight_layout()  # 确保标签不被遮挡

# 显示图表
plt.show()
