import numpy as np
import matplotlib.pyplot as plt

x1, y1, x2, y2 = [], [], [], []


def zhexian():
    txt_path1 = "../logs/joint-angle-avoid_2023-06-20/joint-angle-avoid_20-17-11/progress.txt"
    with open(txt_path1, 'r') as file:
        label1 = []
        for i, line in enumerate(file):
            data_line = line.strip("\n").split()
            if i != 0:
                data_line = [np.float(x) for x in data_line]
            label1.append(data_line)

    for i, value in enumerate(label1):
        if i != 0:
            x1.append(int(value[0]))
            y1.append(value[1])
    plt.plot(x1, y1, color='green', linewidth=1)

    # txt_path2 = "../logs/joint-angle-avoid_2023-06-20/joint-angle-avoid_20-17-11/progress.txt"
    # with open(txt_path2, 'r') as file:
    #     label2 = []
    #     for i, line in enumerate(file):
    #         data_line = line.strip("\n").split()
    #         if i != 0:
    #             data_line = [np.float(x) for x in data_line]
    #         label2.append(data_line)
    # for i, value in enumerate(label2):
    #     if i != 0:
    #         x2.append(int(value[0]))
    #         y2.append(value[12])
    # plt.plot(x2, y2, color='red', linewidth=1)

    # plt.legend()
    # plt.title('Fixed Obstacle Comparison Chart with/without OU-Noise')
    plt.xlabel('Episode')
    plt.ylabel('Average Return')

    plt.show()


def doubleYzhuzhuang():
    plt.rc('font', family='Times New Roman')
    x_data = ['target fixed with OU', 'target fixed without OU', 'target random with OU', 'target random without OU']
    ax_data = [0.97, 0.82, 1.0, 0.94]
    ax1_data = [-23.51, -25.68, -19.77, -21.65]

    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)

    ax.set_ylim([0, 1])
    ax.set_yticks = np.arange(0, 1)
    ax.set_yticklabels = np.arange(0, 1)

    bar_width = 0.3
    ax.set_ylabel('success rate', fontsize=8, fontweight='bold')
    ax.bar(x=np.arange(len(x_data)), width=bar_width, height=ax_data, label='y1', fc='steelblue', alpha=0.8)

    for a, b in enumerate(ax_data):
        plt.text(a, b + 0.0005, '%s' % b, ha='center')

    ax1 = ax.twinx()  # this is the important function

    ax1.set_ylim([-15, -30])
    ax1.set_yticks = np.arange(-15, -30)
    ax1.set_yticklabels = np.arange(-15, -30)
    ax1.set_ylabel('average return', fontsize=8, fontweight='bold')
    ax1.bar(x=np.arange(len(x_data)) + bar_width, width=bar_width, height=ax1_data, label='y2', fc='indianred',
            alpha=0.8)

    for a, b in enumerate(ax1_data):
        plt.text(a + 0.3, b + 0.001, '%s' % b, ha='center')

    plt.xticks(np.arange(len(x_data)) + bar_width / 2, x_data)

    ax.set_xlabel('double Y', fontsize=10, fontweight='bold')

    fig.legend(loc=1, bbox_to_anchor=(0.28, 1), bbox_transform=ax.transAxes)

    plt.show()
    # plt.savefig('your path/name.png')  # 图表输出


def doubleY():
    labels = ['target fixed with OU', 'target fixed without OU', 'target random with OU', 'target random without OU']
    success_rate = [0.97, 0.82, 1.0, 0.94]
    average_return = [-23.51, -25.68, -19.77, -21.65]

    plt.rcParams['axes.labelsize'] = 16  # xy轴label的size
    plt.rcParams['xtick.labelsize'] = 8  # x轴ticks的size
    plt.rcParams['ytick.labelsize'] = 12  # y轴ticks的size
    # plt.rcParams['legend.fontsize'] = 12  # 图例的size

    # 设置柱形的间隔
    width = 0.4  # 柱形的宽度
    x1_list = []
    x2_list = []
    for i in range(len(success_rate)):
        x1_list.append(i)
        x2_list.append(i + width)

    # 创建图层
    fig, ax1 = plt.subplots()

    ax1.spines['top'].set_visible(False)

    # 设置左侧Y轴对应的figure
    ax1.set_ylabel('Success Rate')
    ax1.set_ylim(0, 1)
    ax1.bar(x1_list, success_rate, width=width, color='lightseagreen', align='edge')
    for a, b in enumerate(success_rate):
        plt.text(a+0.19, b + 0.01, '%s' % b, ha='center')

    ax1.set_xticklabels(ax1.get_xticklabels())  # 设置共用的x轴

    # 设置右侧Y轴对应的figure
    ax2 = ax1.twinx()

    ax2.spines['top'].set_visible(False)

    ax2.set_ylabel('Average Reward')
    ax2.set_ylim(-15, -30)
    ax2.bar(x2_list, average_return, width=width, color='tab:blue', align='edge', tick_label=labels)
    for a, b in enumerate(average_return):
        plt.text(a+0.6, b - 0.1, '%s' % b, ha='center')

    ax1.set_xticklabels(labels=labels, rotation=-15)
    plt.tight_layout()
    plt.show()


def doubleYzhexian():
    txt_path1 = "../logs/joint-angle-avoid_2023-06-20/joint-angle-avoid_20-17-11/progress.txt"
    x, y1, y2 = [], [], []
    with open(txt_path1, 'r') as file:
        label1 = []
        for i, line in enumerate(file):
            data_line = line.strip("\n").split()
            if i != 0:
                data_line = [np.float(x) for x in data_line]
            label1.append(data_line)

    for i, value in enumerate(label1):
        if i != 0:
            x.append(int(value[0]))
            y1.append(value[1])
            y2.append(value[12])

    # 创建画布和子图对象
    fig, ax1 = plt.subplots(figsize=(10, 8))

    # 绘制第一个Y轴的折线图
    ax1.plot(x, y1, 'g-', label='average reward')
    ax1.set_xlabel('Episode', fontsize=16)
    ax1.set_ylabel('Average Reward', fontsize=16)
    ax1.tick_params('y')

    # 创建第二个Y轴对象
    ax2 = ax1.twinx()

    # 绘制第二个Y轴的折线图
    ax2.plot(x, y2, 'r-', label='total loss')
    ax2.set_ylabel('Total Loss', fontsize=16)
    ax2.tick_params('y')

    # 添加图例
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels)

    # 显示图像
    plt.show()


# doubleYzhuzhuang()
# doubleY()
# zhexian()
doubleYzhexian()
