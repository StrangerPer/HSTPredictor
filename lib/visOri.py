import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

length = [12]
suffix = ""
for epoch in length:
    suffix = "/M" + str(epoch)
    # 保存的时候，预测和真实值保存反了，所以在这也反
    ground_truth = pd.read_csv("E:/STAEformer-main/vis/PEMS04" + suffix + "_pred.csv", header=None).values.astype(
        np.float32)
    prediction = pd.read_csv("E:/STAEformer-main/vis/PEMS04" + suffix + "_true.csv", header=None).values.astype(
        np.float32)
    Res = pd.read_csv("E:/STAEformer-main/vis/PEMS04" + suffix + "_true.csv", header=None).values.astype(
        np.float32)

    index = 35 # 40 120  PAY-->18 149 P08-->25 35 P04->35 15
    ground_truth = ground_truth[index]
    prediction = prediction[index]

    x = range(0, 576)
    # METRLA需要加上226,平移掉多出来的部分
    # PEMABAY需要加上51
    # PEMS08 110
    # PEMS04 226
    bias = 226
    line4, = plt.plot(x, ground_truth[bias + 576:bias + 1152], '-', label="Groundtruth", color='lightblue')
    line1, = plt.plot(x, prediction[bias + 576:bias + 1152], '--', label="Prediction", color='orange')
    line0, = plt.plot(x, prediction[bias + 576:bias + 1152] - 100, '--', label="Res", color='cyan')
    plt.legend()
    plt.grid(linestyle='-.')
    ax = plt.gca()  # 表明设置图片的各个轴，plt.gcf()表示图片本身
    plt.xticks(np.arange(0, 577, 48),
               ('00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00', '04:00', '08:00', '12:00', '16:00',
                '20:00', '24:00'))
    plt.xticks(rotation=45)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(48))
    plt.savefig("E:/STAEformer-main/fig/PEMS04/M" + str(epoch) + str(index) + ".png", bbox_inches='tight')
    plt.show()
