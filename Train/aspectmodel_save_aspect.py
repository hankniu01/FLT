
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
# from rgat import RGAT
from complex_att import OriMultiHeadAttention, TokDctMultiHeadAttention, TokAutoDctMultiHeadAttention, KernelMultiHeadAttention, \
        ConsineMultiHeadAttention

from fastNLP.embeddings import BertWordPieceEncoder, RobertaWordPieceEncoder


class AspectModel(nn.Module):
    def __init__(self, args, embed, dropout, num_classes, pool="max"):
        super().__init__()
        assert pool in ("max", "mean")
        self.args = args
        self.sparse_hold = args.sparse_hold
        self.attention_heads = self.args.attention_heads
        self.embed = embed
        self.embed_dropout = nn.Dropout(dropout)
        self.num_probe_layers = len(self.args.probe_layers.split(','))
        if hasattr(embed, "embedding_dim"):
            embed_size = embed.embedding_dim
        else:
            embed_size = embed.config.hidden_size
        
      
        if self.args.freq_type == 'ori':
            self.attn = OriMultiHeadAttention(self.attention_heads, embed_size, h_dim=self.args.h_dim, combination=self.args.combination)
        elif self.args.freq_type == 'tkdft':
            self.attn =TokDctMultiHeadAttention(self.args, self.attention_heads, embed_size, h_dim=self.args.h_dim)
        elif self.args.freq_type == 'tkauto':
            self.attn =TokAutoDctMultiHeadAttention(self.args, self.attention_heads, embed_size, h_dim=self.args.h_dim)
        elif self.args.freq_type == 'kernel':
            self.attn =KernelMultiHeadAttention(self.args, self.attention_heads, embed_size, h_dim=self.args.h_dim)
        elif self.args.freq_type == 'w_cosine':
            self.attn =ConsineMultiHeadAttention(self.args, self.attention_heads, embed_size, h_dim=self.args.h_dim)
            
            
        # if self.args.gnn == 'gcn':
        #     self.gc0 = nn.ModuleList([GraphConvolution(embed_size, embed_size) for _ in range(self.args.layers)])
        # elif self.args.gnn == 'rgat':
        #     self.gc0 = nn.ModuleList([RGAT(60) for _ in range(self.args.layers)])
        # elif self.args.gnn == 'attgcn':
        self.gcn_drop = nn.Dropout(0.3)
        self.weight_list = nn.ModuleList()
        for layer in range(self.args.layers):
            self.weight_list.append(nn.Linear(embed_size, embed_size))
        self.weight_list.requires_grad_ = False
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_size, num_classes),
        )
        self.pool = pool

    def forward(self, tokens, aspect_mask):
        """

        :param tokens:
        :param aspect_mask: bsz x max_len, 1 for aspect
        :return:
        """
        
        self.args.step += 1
        if self.args.step % 100 == 0:
            self.args.gumbel_temprature = max( np.exp((self.args.step+1) *-1* self.args.gumbel_decay), .05)
            
        fmask = (torch.ones_like(tokens) != tokens).float()
        mask_ = fmask.unsqueeze(-1)@fmask.unsqueeze(1)
        if isinstance(self.embed, BertWordPieceEncoder):
            tokens = self.embed(tokens, None)  # bsz x max_len x hidden_size
        else:
            tokens, hidden_states = self.embed(
                tokens, token_type_ids=None
            )  # bsz x max_len x hidden_size
        
        if isinstance(tokens, tuple):
            tokens = tokens[0]

        tokens = self.embed_dropout(tokens)

        if self.num_probe_layers > 1:
            tokens_probe = hidden_states[int(self.args.probe_layers.split(',')[-1])]
            tokens_probe = self.embed_dropout(tokens_probe)
        else:
            tokens_probe = tokens
        
        for l in range(self.args.layers):
            # raw_adj, raw_rel = self.graphlearner[i](tokens)
            if self.args.metric_type != 'att':
                raw_adj, raw_rel = self.graphlearner(tokens_probe)
            else:
                attn_tensor = self.attn(tokens_probe, tokens_probe, mask_)
                attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
                adj_ag = None
                # * Average Multi-head Attention matrixes
                for i in range(self.attention_heads):
                    if adj_ag is None:
                        adj_ag = attn_adj_list[i]
                    else:
                        adj_ag = adj_ag + attn_adj_list[i]
                adj_ag = adj_ag / self.attention_heads
                
                for j in range(adj_ag.size(0)):
                    adj_ag[j] = adj_ag[j] - torch.diag(torch.diag(adj_ag[j]))
                    adj_ag[j] = adj_ag[j] + torch.eye(adj_ag[j].size(0)).cuda()
                    # adj_ag = fmask * adj_ag
                raw_adj = adj_ag
            
            if self.args.gnn == 'gcn':
                raw_adj = torch.softmax(raw_adj, dim=-1)
                tokens = F.relu(self.gc0[l](tokens, raw_adj))
            elif self.args.gnn == 'rgat':
                tokens = self.gc0[l](tokens, raw_rel, raw_adj, fmask)
            elif self.args.gnn == 'attgcn':
                denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
                Ax = adj_ag.bmm(tokens)
                AxW = self.weight_list[l](Ax)
                AxW = AxW / denom_ag
                gAxW = F.relu(AxW)
                tokens = self.gcn_drop(gAxW) if l < self.args.layers - 1 else gAxW

        aspect_mask = aspect_mask.eq(1)
        if self.pool == "mean":
            tokens = tokens.masked_fill(aspect_mask.unsqueeze(-1).eq(0), 0)
            tokens = tokens.sum(dim=1)
            preds = tokens / aspect_mask.sum(dim=1, keepdims=True).float()
        elif self.pool == "max":
            aspect_mask = aspect_mask.unsqueeze(-1).eq(0)  # bsz x max_len x 1
            tokens = tokens.masked_fill(aspect_mask, -10000.0)
            preds, _ = tokens.max(dim=1)
        # # for tokens_g
        # if self.pool == "mean":
        #     tokens_g = tokens_g.masked_fill(aspect_mask.unsqueeze(-1).eq(0), 0)
        #     tokens_g = tokens_g.sum(dim=1)
        #     preds_g = tokens_g / aspect_mask.sum(dim=1, keepdims=True).float()
        # elif self.pool == "max":
        #     # aspect_mask = aspect_mask.unsqueeze(-1).eq(0)  # bsz x max_len x 1
        #     tokens_g = tokens_g.masked_fill(aspect_mask, -10000.0)
        #     preds_g, _ = tokens_g.max(dim=1)
        
        # preds = preds + preds_g
        preds_ = self.ffn(preds)
        return preds_, preds





class MlpModel(nn.Module):
    def __init__(self, args, embed, dropout, num_classes, pool="max"):
        super().__init__()
        assert pool in ("max", "mean")
        self.args = args
        self.sparse_hold = args.sparse_hold
        self.attention_heads = self.args.attention_heads
        self.embed = embed
        self.embed_dropout = nn.Dropout(dropout)
        self.num_probe_layers = len(self.args.probe_layers.split(','))
        if hasattr(embed, "embedding_dim"):
            embed_size = embed.embedding_dim
        else:
            embed_size = embed.config.hidden_size
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_size, num_classes),
        )
        self.pool = pool

    def forward(self, tokens, aspect_mask):
        """

        :param tokens:
        :param aspect_mask: bsz x max_len, 1 for aspect
        :return:
        """
        
        self.args.step += 1
        if self.args.step % 100 == 0:
            self.args.gumbel_temprature = max( np.exp((self.args.step+1) *-1* self.args.gumbel_decay), .05)
            
        fmask = (torch.ones_like(tokens) != tokens).float()
        mask_ = fmask.unsqueeze(-1)@fmask.unsqueeze(1)
        if isinstance(self.embed, BertWordPieceEncoder):
            tokens = self.embed(tokens, None)  # bsz x max_len x hidden_size
        else:
            tokens, hidden_states = self.embed(
                tokens, token_type_ids=None
            )  # bsz x max_len x hidden_size
        
        if isinstance(tokens, tuple):
            tokens = tokens[0]

        tokens = self.embed_dropout(tokens)

        aspect_mask = aspect_mask.eq(1)
        if self.pool == "mean":
            tokens = tokens.masked_fill(aspect_mask.unsqueeze(-1).eq(0), 0)
            tokens = tokens.sum(dim=1)
            preds = tokens / aspect_mask.sum(dim=1, keepdims=True).float()
        elif self.pool == "max":
            aspect_mask = aspect_mask.unsqueeze(-1).eq(0)  # bsz x max_len x 1
            tokens = tokens.masked_fill(aspect_mask, -10000.0)
            preds, _ = tokens.max(dim=1)
        # # for tokens_g
        # if self.pool == "mean":
        #     tokens_g = tokens_g.masked_fill(aspect_mask.unsqueeze(-1).eq(0), 0)
        #     tokens_g = tokens_g.sum(dim=1)
        #     preds_g = tokens_g / aspect_mask.sum(dim=1, keepdims=True).float()
        # elif self.pool == "max":
        #     # aspect_mask = aspect_mask.unsqueeze(-1).eq(0)  # bsz x max_len x 1
        #     tokens_g = tokens_g.masked_fill(aspect_mask, -10000.0)
        #     preds_g, _ = tokens_g.max(dim=1)
        
        # preds = preds + preds_g
        preds = self.ffn(preds)
        return {"pred": preds}