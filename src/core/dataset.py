import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class create_Dataset():
    def __init__(self, datasetNmae, labelType):
        DATA_MAP = {
            'emotake': self.__init_emotake,
            'other': None
        }
        data, multi_task, label = DATA_MAP[datasetNmae](labelType)
        self.d_l = {'data': data, 'multi_task': multi_task, 'label': label}

    def __init_emotake(self, labelType):
        # au_data = np.load('data/aus.npy').transpose(0, 2, 1)
        # em_data = np.load('data/ems.npy').transpose(0, 2, 1)
        # hp_data = np.load('data/hps.npy').transpose(0, 2, 1)
        au_data = np.load('data/aus.npy')
        em_data = np.load('data/ems.npy')
        hp_data = np.load('data/hps.npy')

        # Body Poster 的维度是 batch，lenght，point1，point2
        # 300 12 2 中的 12 2 代表的是每个图片提取出了12个点，2是每个点的横纵坐标
        bp_data = np.load('data/bps.npy')
        # bp_data = bp_data.reshape(bp_data.shape[0], bp_data.shape[1], -1).transpose(0, 2, 1)
        bp_data = bp_data.reshape(bp_data.shape[0], bp_data.shape[1], -1)

        combine_data = [{'au': au_data[i], 'em': em_data[i], 'hp': hp_data[i], 'bp': bp_data[i]} for i in range(len(au_data))]

        quality_label = np.load('data/quality.npy', allow_pickle=True)
        quality_label = np.array(quality_label, dtype=int)
        ra_label = np.load('data/ra.npy', allow_pickle=True)
        ra_label = np.array(ra_label, dtype=int)
        readiness_label = np.load('data/readiness.npy', allow_pickle=True)
        readiness_label = np.array(readiness_label, dtype=int)

        multi_task = []
        for i in range(len(quality_label)):
            multi_task.append([quality_label[i], ra_label[i], readiness_label[i]])

        if labelType == 'quality':
            combine_label = quality_label
        elif labelType == 'ra':
            combine_label = ra_label
        else:
            combine_label = readiness_label

        return combine_data, multi_task, combine_label

    def __len__(self):
        return len(self.dataset['label'])


def create_DataLoader(opt, dataset):
    dataset = MDDataset(dataset)
    dataLoader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=True
    )
    return dataLoader


class MDDataset(Dataset):
    def __init__(self, dataset):
        self.reconstruct_dataset(dataset)

    def reconstruct_dataset(self, dataset):
        data, label = dataset[0], dataset[1]
        au, em, hp, bp = [], [], [], []
        for i in range(len(data)):
            au.append(data[i]['au'])
            em.append(data[i]['em'])
            hp.append(data[i]['hp'])
            bp.append(data[i]['bp'])

        self.au = torch.tensor(np.array(au), dtype=torch.float32)
        self.em = torch.tensor(np.array(em), dtype=torch.float32)
        self.hp = torch.tensor(np.array(hp), dtype=torch.float32)
        self.bp = torch.tensor(np.array(bp), dtype=torch.float32)

        self.au_lengths = len(self.au[0])
        self.em_lengths = len(self.em[0])
        self.hp_lengths = len(self.hp[0])
        self.bp_lengths = len(self.bp[0])

        # # Clear dirty data
        # self.audio[self.audio == -np.inf] = 0
        # self.vision[self.vision == -np.inf] = 0
        self.__gen_mask()
        self.label = torch.tensor(label)

    def __gen_mask(self):
        mask_list = []
        for data in [self.au, self.em, self.hp, self.bp]:
            mask = torch.tensor([[False for i in range(data.shape[1])] for j in range(data.shape[0])])
            mask_list.append(mask)
        self.padding_mask_au = mask_list[0]
        self.padding_mask_em = mask_list[1]
        self.padding_mask_hp = mask_list[2]
        self.padding_mask_bp = mask_list[3]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        sample = {
            'au': self.au[index],
            'em': self.em[index],
            'hp': self.hp[index],
            'bp': self.bp[index],
            'au_lengths': self.au_lengths,
            'em_lengths': self.em_lengths,
            'hp_lengths': self.hp_lengths,
            'bp_lengths': self.bp_lengths,
            'padding_mask_au': self.padding_mask_au[index],
            'padding_mask_em': self.padding_mask_em[index],
            'padding_mask_hp': self.padding_mask_hp[index],
            'padding_mask_bp': self.padding_mask_bp[index],
            'label': self.label[index],
            'index': index
        }
        return sample
