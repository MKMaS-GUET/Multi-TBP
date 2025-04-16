import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.EncoderModule import UnimodalEncoder
from models.FusionModual import MultimodalFusion
from models.iTransformer import iTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class TPER(nn.Module):
    def __init__(self, opt):
        super(TPER, self).__init__()
        # Unimodal Embedding and Encoder
        self.UniEncoder = UnimodalEncoder(opt)

        # Multimodal Interaction and Fusion
        self.MultiFusion = MultimodalFusion(opt)

        # Classification Prediction
        self.CLS1 = SentiCLS(fusion_dim=256*8, n_class=3)   # quality
        self.CLS2 = SentiCLS(fusion_dim=256*8, n_class=3)   # ra
        self.CLS3 = SentiCLS(fusion_dim=256*8, n_class=2)   # readiness

    def forward(self, input_data):
        au, em, hp, bp = input_data['bp'], input_data['bp'], input_data['bp'], input_data['bp']
        padding_mask = input_data['padding_mask']
        lengths = input_data['length']

        data_tsne = {}
        for types in ['au', 'em', 'hp', 'bp']:
            data_tsne[types] = torch.mean(input_data[types], dim=1)

        # Unimodal Encoding
        uni_token, uni_utterance = self.UniEncoder(au, em, hp, bp, padding_mask, lengths)

        # Dynamic Multimodal Fusion using High-level Semantic Features
        multi_fea = self.MultiFusion(uni_token)

        data_fusion_tsne = {}
        for i, types in enumerate(['au', 'em', 'hp', 'bp']):
            data_fusion_tsne[types] = torch.mean(multi_fea[:, :, 256*i:256*(i+1)], dim=1)

        # Sentiment Classification
        prediction1 = self.CLS1(uni_utterance, multi_fea)
        prediction2 = self.CLS2(uni_utterance, multi_fea)
        prediction3 = self.CLS3(uni_utterance, multi_fea)

        return {'quality': prediction1, 'ra': prediction2, 'readiness': prediction3}, data_tsne, data_fusion_tsne


class SentiCLS(nn.Module):
    def __init__(self, fusion_dim, n_class, classifier_dropout=0.1):
        super(SentiCLS, self).__init__()
        # self.cls_layer = nn.Sequential(
        #     nn.GELU(),
        #     nn.Dropout(p=classifier_dropout),
        #     nn.Linear(fusion_dim, fusion_dim // 2),
        #     nn.GELU(),
        #     nn.Linear(fusion_dim // 2, n_class)
        # )
        self.linear1 = nn.Linear(fusion_dim, fusion_dim // 2)
        self.linear2 = nn.Linear(fusion_dim // 2, n_class)
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.activate = nn.GELU()

    def forward(self, uni_fea, multi_fea):
        multi_fea = torch.mean(multi_fea, dim=1)
        # for Type in ['au', 'em', 'hp', 'bp']:
        #     multi_fea[Type] = torch.mean(multi_fea[Type], dim=1)

        joint_fea = torch.cat((uni_fea['au'], uni_fea['em'], uni_fea['hp'], uni_fea['bp'], multi_fea), dim=-1)
        # output = self.cls_layer(joint_fea)

        output = self.linear1(self.dropout(self.activate(joint_fea)))
        output = self.linear2(self.activate(output))
        return output


class LinearModel(nn.Module):
    def __init__(self, fea_dim, n_class):
        super(LinearModel, self).__init__()
        self.cls_layer = nn.Sequential(
            nn.Linear(fea_dim, fea_dim // 2),
            # nn.GELU(),
            nn.Linear(fea_dim // 2, n_class)
        )

    def forward(self, uni_fea):
        for types in ['au', 'em', 'hp', 'bp']:
            # uni_fea[types] = uni_fea[types].permute(0, 2, 1)
            uni_fea[types] = torch.mean(uni_fea[types], dim=1)

        joint_fea = torch.cat((uni_fea['au'], uni_fea['em'], uni_fea['hp'], uni_fea['bp']), dim=-1)
        output = self.cls_layer(joint_fea)
        return output


class LSTMModel(nn.Module):
    def __init__(self, fea_dim, layers, n_class):
        super(LSTMModel, self).__init__()
        self.rnn = nn.LSTM(input_size=fea_dim, hidden_size=fea_dim, num_layers=layers, bidirectional=True, dropout=0.1, batch_first=True)
        self.proj_last_state = nn.Linear(2 * fea_dim, n_class)

    def forward(self, uni_fea):
        length = uni_fea['length']['au'] + uni_fea['length']['em'][0] + uni_fea['length']['hp'][0] + uni_fea['length']['bp'][0]
        length = length.to('cpu').to(torch.int64)
        joint_fea = torch.cat((uni_fea['au'], uni_fea['em'], uni_fea['hp'], uni_fea['bp']), dim=1)
        packed_sequence = pack_padded_sequence(joint_fea, length, batch_first=True, enforce_sorted=False)  # input:输入数据 lengths:每条数据本身的长度（padding前）
        packed_h, final_h_c_out = self.rnn(packed_sequence)
        # padded_h, _ = pad_packed_sequence(packed_h, batch_first=True, total_length=length[0])
        # hidden_state = self.proj_hidden(padded_h)

        h_sent_out = final_h_c_out[0]
        h_sent_seq = torch.cat((h_sent_out[0], h_sent_out[1]), dim=-1)
        output = self.proj_last_state(h_sent_seq)
        return output


def build_model(opt):
    model = TPER(opt)
    # model = LinearModel(fea_dim=353, n_class=opt.num_class)
    # model = LSTMModel(fea_dim=300, layers=1, n_class=opt.num_class)
    # model = iTransformer(num_class=opt.num_class)
    # model = KNeighborsClassifier(n_neighbors=3)
    # model = RandomForestClassifier()
    # model = SVC()
    return model
