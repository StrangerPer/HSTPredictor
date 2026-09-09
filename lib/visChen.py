import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ground_truth = pd.read_csv("E:/STAEformer-main/vis/PEMS04/M12" + "_pred.csv", header=None).values.astype(
        np.float32)
prediction = pd.read_csv("E:/STAEformer-main/vis/PEMS04/M12" + "_true.csv", header=None).values.astype(
        np.float32)
# 选取需要进行可视化的节点
index = 35  # 40 120  PAY-->18 149 P08-->25 35 P04->35 15
ground_truth = ground_truth[index]
prediction = prediction[index]
# 原始的
# obs = pd.read_csv('E:/STAEformer-main/data/road1_fiber_ms50_r10.csv')
# data = obs.values
bias = 226
plt.style.use('classic')
fig = plt.figure(figsize=(13, 2.5))
ax = fig.add_axes([0.05, 0.22, 0.92, 0.75])
# 一周数据
plt.plot(ground_truth[bias + 576:bias + 2592], color="#006ea3", linewidth=0.4, label="Actual value")
# plt.plot(data[0:14 * 144 - 1, 2], color="#f79b5f", linewidth=1.8, label=r"Global parameter + biases")
plt.plot(prediction[bias + 576:bias + 2592], color="#e3120b", linewidth=1.0, label="Estimated value")

ax.set_xlim([0, 14 * 144 - 1])
ax.set_ylim([0, 800])   # 设置y轴的最大值
ax.grid(color='gray', linestyle='-', linewidth=0.1, alpha=0.2)
ax.grid(axis='x', linestyle='--', linewidth=2)
ax.grid(axis='y', linestyle='-', linewidth=1)

for i in range(14):
    # if data[144 * i, 1] > 0:
    # 给图片添加绿色背景
    someX, someY = i * 144, 0
    currentAxis = plt.gca()
    ax.add_patch(patches.Rectangle((someX, someY), 144, 800,
                                   alpha=0.1, facecolor='green'))   # 修改最大值时，这里也记得修改

plt.xticks(np.arange(0, 14 * 144, 72), ["00:00", "12:00", "00:00", "12:00",
                                        "00:00", "12:00", "00:00", "12:00",
                                        "00:00", "12:00", "00:00", "12:00",
                                        "00:00", "12:00", "00:00", "12:00",
                                        "00:00", "12:00", "00:00", "12:00",
                                        "00:00", "12:00", "00:00", "12:00",
                                        "00:00", "12:00", "00:00", "12:00"], rotation=30)
plt.yticks(np.arange(100, 800, 100), [100, 200, 300, 400, 500, 600, 700])
ax.set_ylabel("Speed (km/h)")
plt.legend(ncol=1, loc=4)  # 控制标签的位置点
plt.show()
# fig.savefig("time_series_speed1.pdf")
