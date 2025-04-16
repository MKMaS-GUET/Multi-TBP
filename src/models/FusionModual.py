import copy
import torch
from torch import nn, einsum
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            # nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.net(x)


class MultiHAtten(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(MultiHAtten, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class CrossTransformer(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super(CrossTransformer, self).__init__()
        self.cross_attn = MultiHAtten(dim, heads=8, dim_head=64, dropout=dropout)
        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, target_x, source_x):
        target_x_tmp = self.cross_attn(target_x, source_x, source_x)
        target_x = self.layernorm1(target_x_tmp + target_x)
        target_x = self.layernorm2(self.ffn(target_x) + target_x)
        return target_x


class MultimodalFusion(nn.Module):
    def __init__(self, opt):
        super(MultimodalFusion, self).__init__()
        # Length Align
        # self.len_au = nn.Linear(opt.seq_lens[0], 8)
        # self.len_em = nn.Linear(opt.seq_lens[1], 8)
        # self.len_hp = nn.Linear(opt.seq_lens[2], 8)
        # self.len_bp = nn.Linear(opt.seq_lens[3], 8)
        self.len_au = nn.Linear(300, 16)
        self.len_em = nn.Linear(300, 16)
        self.len_hp = nn.Linear(300, 16)
        self.len_bp = nn.Linear(300, 16)

        # 初始化迭代融合模块
        # fusion_block = DynFusBlock(opt)
        # self.dec_list = self._get_clones(fusion_block, 3)
        self.crotras1 = CrossTransformer(dim=1024, mlp_dim=1024, dropout=0.1)
        self.crotras2 = CrossTransformer(dim=1024, mlp_dim=1024, dropout=0.1)
        self.crotras3 = CrossTransformer(dim=1024, mlp_dim=1024, dropout=0.1)

    def forward(self, uni_fea, uni_mask=None):
        hidden_au = self.len_au(uni_fea['au'].permute(0, 2, 1)).permute(0, 2, 1)
        hidden_em = self.len_em(uni_fea['em'].permute(0, 2, 1)).permute(0, 2, 1)
        hidden_hp = self.len_hp(uni_fea['hp'].permute(0, 2, 1)).permute(0, 2, 1)
        hidden_bp = self.len_bp(uni_fea['bp'].permute(0, 2, 1)).permute(0, 2, 1)
        target = torch.cat((hidden_au, hidden_em, hidden_hp, hidden_bp), dim=-1)

        target = self.crotras1(target, target)
        target = self.crotras2(target, target)
        target = self.crotras3(target, target)

        # source = hidden_au  # 进行渐进式融合，首先将其中一个模态作为融合模块的输入，即第一个融合源数据
        # other_hidden = [hidden_em, hidden_hp, hidden_bp]
        # for i, dec in enumerate(self.dec_list):
        #     source = dec(source, other_hidden[i], uni_mask)     # 接着后续的所有模态依次融合，寻找模态间相似性的特征

        return target

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
