import os
import logging
import random
import numpy as np
import torch
from tabulate import tabulate
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.ticker as ticker
import pandas as pd


class AverageMeter(object):
    def __init__(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def update(self, value, count):
        self.value = value
        self.value_sum += value * count
        self.count += count
        self.value_avg = self.value_sum / self.count


def ConfigLogging(file_path):
    # 创建一个 logger
    logger = logging.getLogger("save_option_results")
    logger.setLevel(logging.DEBUG)

    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(filename=file_path, encoding='utf8')
    fh.setLevel(logging.DEBUG)

    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_model(save_path, result, modality, model):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_path = os.path.join(
        save_path,
        'MOSEI_{}_MAE-{}_Corr-{}.pth'.format(
            modality,
            result["MAE"],
            result["Corr"]
        )
    )
    torch.save(model.state_dict(), save_file_path)


def save_print_results(opt, logger, train_re, test_re):
    if opt.datasetName in ['emotake']:
        results = [
            ["Train", train_re["Accuracy"], train_re["F1-Score"]],
            ["Test", test_re["Accuracy"], test_re["F1-Score"]]
        ]
        headers = ["Phase", "Accuracy", "F1-Score"]

        table = '\n' + tabulate(results, headers, tablefmt="grid") + '\n'
        logger.info(table)
    else:
        results = [
            ["Train", train_re["MAE"], train_re["Corr"], train_re["Mult_acc_2"], train_re["Mult_acc_3"], train_re["Mult_acc_5"], train_re["F1_score"]],
            ["Test", test_re["MAE"], test_re["Corr"], test_re["Mult_acc_2"], test_re["Mult_acc_3"], test_re["Mult_acc_5"], test_re["F1_score"]]
        ]
        headers = ["Phase", "MAE", "Corr", "Acc-2", "Acc-3", "Acc-5", "F1"]

        table = '\n' + tabulate(results, headers, tablefmt="grid") + '\n'
        if logger is not None:
            logger.info(table)
        else:
            print(table)


def calculate_u_test(pred, label):
    pred, label = pred.squeeze().numpy(), label.squeeze().numpy()
    label_mean = np.mean(label)
    alpha = 0.05

    pred_mean = np.mean(pred)
    pred_std = np.std(pred, ddof=1)
    label_std = np.std(label, ddof=1)
    # standard_error = pred_std / np.sqrt(len(pred))
    standard_error = np.sqrt(label_std / len(label) + pred_std / len(pred))

    Z = (label_mean - pred_mean) / standard_error
    critical_value = stats.norm.ppf(1 - alpha)
    if Z >= critical_value:
        print("拒绝原假设，接受备择假设")
    else:
        print("无法拒绝原假设")


def plot_tsne(data, fusion, path):
    def data_visual(data_lowDWeights, true_label, names):
        true_label = true_label.reshape((-1, 1))
        all_data = np.hstack((data_lowDWeights, true_label))
        all_data = pd.DataFrame({
            'x': all_data[:, 0],
            'y': all_data[:, 1],
            'label': all_data[:, 2]
        })

        for i, index in enumerate(label_com):
            X = data_lowDWeights[i*39:(i+1)*39, 0]
            Y = data_lowDWeights[i*39:(i+1)*39, 1]
            Z = data_lowDWeights[i*39:(i+1)*39, 2]
            ax.scatter(X, Y, Z, c=colors[index], s=15, marker=makers[index], alpha=1, label=label_shape[i])

        # 调整图像角度
        ax.view_init(elev=25, azim=50)

        # 图例位置
        # ax.legend(loc='upper right', prop=font, ncol=4, handlelength=1, borderpad=0.8, labelspacing=0.2)

        # 设置坐标系刻度大小
        # ax.tick_params(labelsize=10)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])

        # 设置坐标系范围
        # ax.set_xlim3d([-1000, 1000])
        # ax.set_ylim3d([-1000, 1000])
        # ax.set_zlim3d([-1000, 1000])
        # ax.tick_params(pad=-3)

    input, input_f = [], []
    for i in ['au', 'em', 'hp', 'bp']:
        m = torch.cat((data[0][i], data[1][i]), dim=0)
        if m.shape[1] <= 10:
            m = torch.nn.functional.pad(m, (0, int(10-m.shape[1]), 0, 0), mode='constant', value=0)
        else:
            m = m[:, 0:10]
        input.append(m)
        n = torch.cat((fusion[0][i], fusion[1][i]), dim=0)
        if n.shape[1] <= 10:
            n = torch.nn.functional.pad(n, (0, int(10-n.shape[1]), 0, 0), mode='constant', value=0)
        else:
            n = n[:, 0:10]
        input_f.append(n)
    input = torch.cat(input).cpu()
    input_f = torch.cat(input_f).cpu()
    input = torch.cat((input, input_f), dim=0)

    # label_com = ['hidden_au', 'hidden_em', 'hidden_hp', 'hidden_bp']    # 图例名称
    label_com = ['hidden_au', 'hidden_em', 'hidden_hp', 'hidden_bp', 'fusion_au', 'fusion_em', 'fusion_hp', 'fusion_bp']    # 图例名称
    label_shape = [
        r'$h_{au}$',
        r'$h_{em}$',
        r'$h_{hp}$',
        r'$h_{bp}$',
        r'$f_{au}$',
        r'$f_{em}$',
        r'$f_{hp}$',
        r'$f_{bp}$'
    ]    # 图例名称
    label = []
    for y in label_com:
        for i in range(39):
            label.append(y)
    label = np.array(label)

    # color_ratio = data['R']
    # data = torch.cat([data['T'], data['V'], data['A'], data['F']], dim=0)
    # data = data.cpu().detach().numpy()

    makers = {
        'hidden_au': '.',
        'hidden_em': '.',
        'hidden_hp': '.',
        'hidden_bp': '.',
        'fusion_au': '^',
        'fusion_em': '^',
        'fusion_hp': '^',
        'fusion_bp': '^'
    }  # 设置散点形状

    colors = {
        'hidden_au': ['#CC3399'],   # 玫红色，说明面部表情(情绪)是主要引导寻找相似特征的向量
        'hidden_em': ['#009900'],
        'hidden_hp': ['#FF9933'],
        'hidden_bp': ['#4472C4'],
        'fusion_au': ['#CC3399'],
        'fusion_em': ['#009900'],
        'fusion_hp': ['#FF9933'],
        'fusion_bp': ['#4472C4']
    }    # 设置散点颜色

    font = {
        'weight': 'normal',
        'size': 5
    }   # 设置字体

    tsne = TSNE(n_components=3, init='pca', random_state=0)
    data_ts = tsne.fit_transform(input)
    np.save("t-sne_data_fusion_linear.npy", data_ts)
    # data_ts = data

    # fig = plt.figure(figsize=(10, 10))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    data_visual(data_ts, label, path)
    fig.savefig('/opt/data/private/Project/Multi-TPER/data/t-sne-fusion-linear.png', dpi=600, bbox_inches='tight')
