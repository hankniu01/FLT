import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import torch.autograd as autograd

def attention(query, key, mask=None, dropout=None, combination='multi'):
    '''
    query: query_r, query_i   B*L*H*R, where R is rank
    key = (key_r, key_i)
    '''
    d_k = query[0].size(-1)
    # scores_rr = torch.matmul(query[0], key[0].transpose(-2, -1)) / math.sqrt(d_k)
    # scores_ii = torch.matmul(query[1], key[1].transpose(-2, -1)) / math.sqrt(d_k)
    scores_ri = torch.matmul(query[0], key[1].transpose(-2, -1)) / math.sqrt(d_k)
    # scores_ir = torch.matmul(query[1], key[0].transpose(-2, -1)) / math.sqrt(d_k)
    
    scores = scores_ri
    
    # if combination == 'multi':
    #     scores = scores_rr - scores_ii + scores_ri + scores_ir
    # elif combination == 'multi_conj':
    #     scores = scores_rr + scores_ii + scores_ri - scores_ir
    
    
    
    # scores = scores_rr - scores_ii + scores_ri + scores_ir
    # scores = torch.max(torch.stack([scores_rr - scores_ii, scores_ri + scores_ir], -1), -1)[0]   #perform bad
    
    # scores = rel_corr(query, key)
    # scores = torch.sum(scores, -1)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class OriMultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, h_dim=120, dropout=0.1, combination='multi'):
        super(OriMultiHeadAttention, self).__init__()
        assert d_model % h == 0
        assert d_model % (2*h) == 0

        self.comb = combination
        # self.d_k = d_model // h
        self.rank = d_model // 1
        self.d_k = h_dim // h
        self.h = h
        self.r_linears = clones(nn.Linear(self.rank, h_dim), 2)
        self.i_linears = clones(nn.Linear(self.rank, h_dim), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask[:, :, :query.size(1)]
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)
        # max_len = query.shape[1]
        query_r, query_i = query, query                # B*L*D
        key_r, key_i = key, key
        
        # query_r, query_i = query[:, :, self.rank:], query[:, :, :self.rank]                # B*L*D
        # key_r, key_i = key[:, :, self.rank:], key[:, :, :self.rank]
        
        query_r, key_r = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.r_linears, (query_r, key_r))]
        
        query_i, key_i = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.i_linears, (query_i, key_i))]
        
        query = (query_r, query_i)
        key = (key_r, key_i)
        # query = query.unsqueeze(-3).repeat(1,1,max_len,1,1) 
        # key = key.unsqueeze(-2).repeat(1,1,1,max_len,1)
        
        attn = attention(query, key, mask=mask, dropout=self.dropout, combination=self.comb)

        return attn


class ConsineMultiHeadAttention(nn.Module):

    def __init__(self, args, h, d_model, h_dim=120, dropout=0.1):
        super(ConsineMultiHeadAttention, self).__init__()
        self.args = args
        assert d_model % h == 0
        assert d_model % (2*h) == 0
        
        # self.d_k = d_model // h
        # self.rank = d_model // 2
        # self.pass_type = args.pass_type
        # if self.pass_type == 'bond':
        #     self.start_freq = args.start_freq
        self.q_pass = args.q_pass
        self.k_pass = args.k_pass
        self.d_k = h_dim * h
        self.h_dim = h_dim
        self.h = h
        self.q_linear = nn.Linear(d_model, self.d_k)
        self.k_linear = nn.Linear(d_model, self.d_k)
        self.dropout = nn.Dropout(p=dropout)

        self.weight_tensor = torch.Tensor(h, self.d_k)
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask[:, :, :query.size(1)]
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)

        context = self.q_linear(query).view(nbatches, -1, self.h_dim)
        
        expand_weight_tensor = self.weight_tensor.unsqueeze(1)
        if len(context.shape) == 3:
            expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

        context_fc = context.unsqueeze(0) * expand_weight_tensor
        context_norm = F.normalize(context_fc, p=2, dim=-1)
        scores = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        return p_attn


class KernelMultiHeadAttention(nn.Module):

    def __init__(self, args, h, d_model, h_dim=120, dropout=0.1):
        super(KernelMultiHeadAttention, self).__init__()
        self.args = args
        assert d_model % h == 0
        assert d_model % (2*h) == 0
        
        # self.d_k = d_model // h
        # self.rank = d_model // 2
        # self.pass_type = args.pass_type
        # if self.pass_type == 'bond':
        #     self.start_freq = args.start_freq
        self.q_pass = args.q_pass
        self.k_pass = args.k_pass
        self.d_k = h_dim * h
        self.h_dim = h_dim
        self.h = h
        self.q_linear = nn.Linear(d_model, self.d_k)
        self.k_linear = nn.Linear(d_model, self.d_k)
        self.dropout = nn.Dropout(p=dropout)

        self.precision_inv_dis = nn.Parameter(torch.Tensor(1, 1))
        self.precision_inv_dis.data.uniform_(0, 1.0)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(self.d_k, self.d_k)))
    

    def compute_distance_mat(self, X, weight=None):
        if weight is not None:
            trans_X = X @ weight
        else:
            trans_X = X
        norm = torch.sum(trans_X * X, dim=-1)
        dists = -2 * torch.bmm(trans_X, X.transpose(-1, -2)) + norm.unsqueeze(-1) + norm.unsqueeze(1)
        return dists


    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask[:, :, :query.size(1)]
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)

        context = self.q_linear(query).view(nbatches, -1, self.h_dim)
        
        dist_weight = torch.mm(self.weight, self.weight.transpose(-1, -2))
        attention = self.compute_distance_mat(context, dist_weight)
        scores = torch.exp(-0.5 * attention * (self.precision_inv_dis**2))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        return p_attn
    

def gumbel_softmax(input, temperature):
    noise = torch.rand(input.size()).to(input.device)
    noise.add_(1e-9).log_().neg_()
    noise.add_(1e-9).log_().neg_()
    noise = autograd.Variable(noise)
    x = (input + noise) / temperature
    x = F.softmax(x.view(-1,  x.size()[-1]), dim=-1)
    return x.view_as(input)

        
class TokDctMultiHeadAttention(nn.Module):
    
    def __init__(self, args, h, d_model, h_dim=120, dropout=0.1):
        super(TokDctMultiHeadAttention, self).__init__()
        self.args = args
        assert d_model % h == 0
        assert d_model % (2*h) == 0
        
        # self.d_k = d_model // h
        # self.rank = d_model // 2
        self.pass_type = args.pass_type
        if self.pass_type == 'bond' or self.pass_type == 'high_bond':
            self.start_freq = args.start_freq
        self.q_pass = args.q_pass
        self.k_pass = args.k_pass
        self.bond_q_pass = args.bond_q_pass
        self.bond_k_pass = args.bond_k_pass
        self.d_k = h_dim * h
        self.h_dim = h_dim
        self.h = h
        self.q_linear = nn.Linear(d_model, self.d_k)
        self.k_linear = nn.Linear(d_model, self.d_k)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask[:, :, :query.size(1)]
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)
        # max_len = query.shape[1]
        
        query = self.q_linear(query).view(nbatches, -1, self.h, self.h_dim).permute(0, 2, 3, 1)
        key = self.k_linear(key).view(nbatches, -1, self.h, self.h_dim).permute(0, 2, 3, 1)

        L = query.size(-1)
        d_k = query.shape[2]
        
        q_tff = torch.fft.rfft(query, dim=-1)
        k_tff = torch.fft.rfft(key, dim=-1)

        
        q_mask, k_mask = torch.zeros_like(q_tff), torch.zeros_like(k_tff)
        
        ##### high pass
        if self.pass_type == 'high':
            q_mask[:, :, :, -self.q_pass:] = 1   # low -> high
            k_mask[:, :, :, -self.k_pass:] = 1
        elif self.pass_type == 'low':
            q_mask[:, :, :, :self.q_pass] = 1
            k_mask[:, :, :, :self.k_pass] = 1
        elif self.pass_type == 'high_bond':
            q_mask[:, :, :, -self.q_pass:] = 1   # low -> high
            k_mask[:, :, :, -self.k_pass:] = 1
            q_mask[:, :, :, self.start_freq: self.start_freq + self.bond_q_pass] = 1
            k_mask[:, :, :, self.start_freq: self.start_freq + self.bond_k_pass] = 1
        else: # bond pass
            q_mask[:, :, :, self.start_freq: self.start_freq + self.q_pass] = 1
            k_mask[:, :, :, self.start_freq: self.start_freq + self.k_pass] = 1


        q_tff_flt = q_tff * q_mask
        k_tff_flt = k_tff * k_mask
        
        q = torch.fft.irfft(q_tff_flt, n=L, dim=-1)
        k = torch.fft.irfft(k_tff_flt, n=L, dim=-1)
        
        # scores = q @ k.transpose(2, 3)
        scores = torch.matmul(q.transpose(-2, -1), k) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        return p_attn


class FreqSelector(nn.Module):
    def __init__(self, args, in_dim=1, h_dim=20, dropout=0.1):
        super(FreqSelector, self).__init__()
        self.args = args
        z_dim = 2
        self.dropout = nn.Dropout(p=dropout)
        self.hidden1 = nn.Linear(in_dim, h_dim)
        self.hidden2 = nn.Linear(h_dim, z_dim)
        self.mask = nn.Parameter(torch.zeros(h_dim+1, in_dim))
        
        self.weight_init()
    
    def weight_init(self):
        # nn.init.xavier_normal_(self.hidden1.weight)
        # nn.init.xavier_normal_(self.hidden2.weight)
        nn.init.xavier_uniform_(self.mask)
        
    def gumbel_softmax(self, input, temperature):
        noise = torch.rand(input.size()).to(input.device)
        noise.add_(1e-9).log_().neg_()
        noise.add_(1e-9).log_().neg_()
        noise = autograd.Variable(noise)
        x = (input + noise) / temperature
        x = F.softmax(x.view(-1,  x.size()[-1]), dim=-1)
        # x = F.tanh(x.view(-1,  x.size()[-1])) + 1
        return x.view_as(input)
    
    def forward(self, x):
        '''
        x: B * H * L * D 
        The Freq dimension is D, to select over D
        '''
        B, H, L, D = x.shape
        # h = self.dropout(h)
        # x = self.mask
        x = self.mask
        
        # h = F.relu(torch.view_as_real(self.hidden1(x.transpose(-2, -1))).view(B, H, D, -1))
        h = F.relu(self.hidden1(x))
        logit = self.hidden2(h)

        probes = self.gumbel_softmax(logit, self.args.gumbel_temprature)

        z = probes[:, 1]
        # z = torch.softmax(x, dim=-1)
        
        if self.training:
            z_mask = z
        else:
            z_mask = (z > self.args.dist_threshold).float()
        # z_mask = z
        
        return z_mask.view(1, 1, 1, -1)
    


class TokFreqSelector(nn.Module):
    def __init__(self, args, in_dim=1, h_dim=20, dropout=0.1):
        super(TokFreqSelector, self).__init__()
        self.args = args
        z_dim = 2
        self.dropout = nn.Dropout(p=dropout)
        self.hidden1 = nn.Linear(in_dim, h_dim)
        self.hidden2 = nn.Linear(h_dim, z_dim)
        
    def gumbel_softmax(self, input, temperature):
        noise = torch.rand(input.size()).to(input.device)
        noise.add_(1e-9).log_().neg_()
        noise.add_(1e-9).log_().neg_()
        noise = autograd.Variable(noise)
        x = (input + noise) / temperature
        x = F.softmax(x, dim=-1)
        # x = F.tanh(x.view(-1,  x.size()[-1])) + 1
        return x.view_as(input)
    
    def forward(self, x):
        '''
        x: B * H * L * D 
        The Freq dimension is D, to select over D
        '''
        x = torch.view_as_real(x.transpose(-1, -2))[:, :, :, :, 0]
        # h = self.dropout(h)
        # x = self.mask
               
        # h = F.relu(torch.view_as_real(self.hidden1(x.transpose(-2, -1))).view(B, H, D, -1))
        h = F.relu(self.hidden1(x))
        logit = self.hidden2(h)

        probes = self.gumbel_softmax(logit, self.args.gumbel_temprature)

        z = probes[:, :, :, 1]
        # z = torch.softmax(x, dim=-1)
        
        if self.training:
            # z_mask = (z > self.args.dist_threshold).float()
            z_mask = z
        else:
            z_mask = (z > self.args.dist_threshold).float()
            # z_mask = z
        
        return z_mask.unsqueeze(-2)
    


class TokAutoDctMultiHeadAttention(nn.Module):
    
    def __init__(self, args, h, d_model, h_dim=120, dropout=0.1):
        super(TokAutoDctMultiHeadAttention, self).__init__()
        self.args = args
        assert d_model % h == 0
        assert d_model % (2*h) == 0
        
        # self.d_k = d_model // h
        # self.rank = d_model // 2
        self.n_pass = args.n_pass
        self.d_k = h_dim * h
        self.h_dim = h_dim
        self.h = h
        self.q_linear = nn.Linear(d_model, self.d_k)
        self.k_linear = nn.Linear(d_model, self.d_k)
        
        self.dropout = nn.Dropout(p=dropout)
        self.q_auto_selector = TokFreqSelector(args, in_dim=self.d_k, h_dim=self.d_k//2)
        self.k_auto_selector = TokFreqSelector(args, in_dim=self.d_k, h_dim=self.d_k//2)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask[:, :, :query.size(1)]
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)
        # max_len = query.shape[1]
        
        query = self.q_linear(query).view(nbatches, -1, self.h, self.h_dim).permute(0, 2, 3, 1)
        key = self.k_linear(key).view(nbatches, -1, self.h, self.h_dim).permute(0, 2, 3, 1)

        L = query.size(-1)
        d_k = query.shape[2]
        
        q_tff = torch.fft.rfft(query, dim=-1)
        k_tff = torch.fft.rfft(key, dim=-1)
        
        ######### Auto Selection Frequency #######
        q_mask = self.q_auto_selector(q_tff)
        k_mask = self.k_auto_selector(k_tff)
        
        ############ metric learning ############
        q_tff_flt = q_tff * q_mask
        k_tff_flt = k_tff * k_mask
        
        q = torch.fft.irfft(q_tff_flt, n=L, dim=-1)
        k = torch.fft.irfft(k_tff_flt, n=L, dim=-1)
        
        # scores = q @ k.transpose(2, 3)
        scores = torch.matmul(q.transpose(-2, -1), k) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        return p_attn
    

