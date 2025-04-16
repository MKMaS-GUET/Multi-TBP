import copy
import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PositionEncoding(nn.Module):
    """Construct the CLS token, position and patch embeddings.
    """
    def __init__(self, num_patches, fea_size, tf_hidden_dim, drop_out):
        super(PositionEncoding, self).__init__()
        # self.cls_token = nn.parameter.Parameter(torch.ones(1, 1, tf_hidden_dim))
        self.proj = nn.Linear(fea_size, tf_hidden_dim)
        # self.position_embeddings = nn.parameter.Parameter(torch.zeros(1, num_patches, tf_hidden_dim))
        # self.dropout = nn.Dropout(drop_out)

    def forward(self, embeddings):
        # batch_size = embeddings.shape[0]
        embeddings = self.proj(embeddings)
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        # embeddings = embeddings + self.position_embeddings
        # embeddings = self.dropout(embeddings)
        return embeddings


class TransforEncoderBlock(nn.Module):
    def __init__(self, fea_size, num_patches, nhead, dim_feedforward, num_layers, pos_dropout=0., tf_dropout=0.5):
        super(TransforEncoderBlock, self).__init__()
        self.pos_encoder = PositionEncoding(
            num_patches=num_patches,
            fea_size=fea_size,
            tf_hidden_dim=dim_feedforward,
            drop_out=pos_dropout
        )

        tfencoder_layer = TransformerEncoderLayer(dim_feedforward, nhead, dim_feedforward // 2, dropout=tf_dropout, activation='gelu', batch_first=True)
        self.tfencoder = TransformerEncoder(tfencoder_layer, num_layers)

    def forward(self, src, src_key_padding_mask):
        src = self.pos_encoder(src)
        output = self.tfencoder(src, mask=None, src_key_padding_mask=src_key_padding_mask)
        return output


class TransforEncoder(nn.Module):
    def __init__(self, modality, num_patches, fea_size, dim_feedforward, nhead=2, num_layers=2):
        super(TransforEncoder, self).__init__()
        self.tfencoder = TransforEncoderBlock(
            fea_size=fea_size,
            num_patches=num_patches,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers
        )
        self.layernorm = nn.LayerNorm(dim_feedforward)

    def forward(self, inputs, key_padding_mask):
        tf_last_hidden_state = self.tfencoder(inputs, src_key_padding_mask=key_padding_mask)

        # TransformerEncoder提取的领域知识
        spci_know = self.layernorm(tf_last_hidden_state)

        # 对不同知识进行整合
        uni_hidden = torch.cat([spci_know], dim=-1)
        return uni_hidden

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TemplateEncoder(nn.Module):
    def __init__(self, modality, num_layers, fea_size, dim_feedforward):
        super(TemplateEncoder, self).__init__()
        self.rnn = nn.LSTM(input_size=fea_size, hidden_size=dim_feedforward, num_layers=num_layers, bidirectional=True, dropout=0.1, batch_first=True)
        self.proj_hidden = nn.Linear(2 * dim_feedforward, dim_feedforward)
        self.proj_last_state = nn.Linear(2 * dim_feedforward, dim_feedforward)

    def forward(self, inputs, length):
        '''batch中各个数据的维度是相同的
        为了高效处理, 就需要对样本进行填充(pad_sequence), 使得batch中的各个数据的长度相同, 例如输入的数据x;
        填充之后的样本序列，虽然保证长度相同，但是序列里面可能 padding 很多无效的 0 值，将无效的 padding 值喂给模型进行 forward 会影响模型的效果;
        因此将数据进行padding之后, 送入模型之前, 需要采用 pack_padded_sequence 进行数据压缩(生成 PackedSequence 类), 压缩掉无效的填充值; (让padding值不起作用);
        序列经过模型输出后仍然是压缩序列, 需要使用 pad_packed_sequence 进行解压缩, 就是把原序列填充回来;
        '''
        length = length.to('cpu').to(torch.int64)
        packed_sequence = pack_padded_sequence(inputs, length, batch_first=True, enforce_sorted=False)  # input:输入数据 lengths:每条数据本身的长度（padding前）
        packed_h, final_h_c_out = self.rnn(packed_sequence)
        padded_h, _ = pad_packed_sequence(packed_h, batch_first=True, total_length=length[0])
        hidden_state = self.proj_hidden(padded_h)

        h_sent_out = final_h_c_out[0]
        h_sent_seq = torch.cat((h_sent_out[0], h_sent_out[1]), dim=-1)
        last_state = self.proj_last_state(h_sent_seq)

        return hidden_state, last_state


class CnnEncoder(nn.Module):
    def __init__(self, modality, kernel, fea_size, dim_feedforward):
        super(CnnEncoder, self).__init__()

    def forward(self, inputs, length):
        return None


class UnimodalEncoder(nn.Module):
    def __init__(self, opt):
        super(UnimodalEncoder, self).__init__()
        # Transformer Encoders for learning high-level semantic features
        # self.enc_au = TransforEncoder(modality="au", num_patches=opt.seq_lens[0], fea_size=300, dim_feedforward=512)
        # self.enc_em = TransforEncoder(modality="em", num_patches=opt.seq_lens[1], fea_size=300, dim_feedforward=512)
        # self.enc_hp = TransforEncoder(modality="hp", num_patches=opt.seq_lens[2], fea_size=300, dim_feedforward=512)
        # self.enc_bp = TransforEncoder(modality="bp", num_patches=opt.seq_lens[3], fea_size=300, dim_feedforward=512)

        # LSTM Encoder for learning sequential features
        self.enc_au = TemplateEncoder(modality="au", num_layers=1, fea_size=24, dim_feedforward=256)
        self.enc_em = TemplateEncoder(modality="em", num_layers=1, fea_size=24, dim_feedforward=256)
        self.enc_hp = TemplateEncoder(modality="hp", num_layers=1, fea_size=24, dim_feedforward=256)
        self.enc_bp = TemplateEncoder(modality="bp", num_layers=1, fea_size=24, dim_feedforward=256)

    def forward(self, au, em, hp, bp, mask, length):
        hidden_au, last_au = self.enc_au(au, length['bp'])
        hidden_em, last_em = self.enc_em(em, length['bp'])
        hidden_hp, last_hp = self.enc_hp(hp, length['bp'])
        hidden_bp, last_bp = self.enc_bp(bp, length['bp'])

        hidden_status = {
            'au': hidden_au,
            'em': hidden_em,
            'hp': hidden_hp,
            'bp': hidden_bp
        }
        last_status = {
            'au': last_au,
            'em': last_em,
            'hp': last_hp,
            'bp': last_bp
        }

        # 获取每个模态token-level以及utterance-level的时间序列特征，一方面用于后续跨模态融合，另一方面直接用于映射到联合表征空间
        return hidden_status, last_status
