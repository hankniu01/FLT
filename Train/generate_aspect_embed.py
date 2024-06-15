import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
import warnings
import numpy as np
import json
import pickle

import fitlog
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm
warnings.filterwarnings("ignore")
from fastNLP import (AccuracyMetric, BucketSampler, ClassifyFPreRecMetric,
                     ConstantTokenNumSampler, CrossEntropyLoss, DataSetIter,
                     FitlogCallback, LossBase, RandomSampler,
                     SequentialSampler, SortedSampler, Trainer, WarmupCallback,
                     cache_results)
from fastNLP.core.utils import (_get_model_device, _move_dict_value_to_device,
                                _move_model_to_device)
from fastNLP.embeddings import BertWordPieceEncoder, RobertaWordPieceEncoder
from transformers import XLMRobertaModel, XLNetModel

from ec_pipe import DataPipe

from aspectmodel_save_aspect import AspectModel, MlpModel

from fastNLP.modules.tokenizer import BertTokenizer, RobertaTokenizer

# fitlog.debug()
root_fp = r"/nfsfile/niuhao/project/aspect_process_for_thesis/FLT/Train"
os.makedirs(f"{root_fp}/FT_logs", exist_ok=True)
fitlog.set_log_dir(f"{root_fp}/FT_logs")

def set_seed(seed):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="ec",
    choices=[
        "Restaurants",
        "Laptop",
        "Tweets",
        "alldataset"
        "ec",
        "1516"
    ],
)
parser.add_argument(
    "--data_dir",
    type=str,
    help="dataset dir, should concat with dataset arguement",
)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument(
    "--model_name",
    type=str,
    default="bert-en-base-uncased",
    choices=[
        "bert-en-base-uncased",
        "roberta-en",
        "roberta-en-large",
        "xlmroberta-xlm-roberta-base",
        "bert-multi-base-cased",
        "xlnet-xlnet-base-cased",
    ],
)
parser.add_argument("--save_embed", default=1, type=int)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument('--multi_gpus', default=False, type=bool)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gnn', default='attgcn',
                    choices=['rgat', 'gcn', 'attgcn'])
parser.add_argument('--layers', default=1, type=int)
parser.add_argument('--metric_type', default='att', type=str) # defaukt= att 
parser.add_argument('--attention_heads', default=1, type=int)   # 4
parser.add_argument('--h_dim', default=60, type=int)  # 120   60
parser.add_argument('--combination', default='multi_conj', type=str, choices='multi, multi_conj, div', help='combination for att')

parser.add_argument('--n_components', default=20, type=int)   # 60

parser.add_argument('--gumbel_temprature', default=1, type=int)
parser.add_argument('--gumbel_decay', default=1e-05, type=float)
parser.add_argument('--dist_threshold', default=0.95, type=float)

parser.add_argument('--pass_type', default='high_bond', type=str, choices=['high', 'low', 'bond', 'high_bond', 'mid_high', 'mid_low'])
parser.add_argument('--start_freq', default=5, type=int, help='only implement when pass_type == bond')
parser.add_argument('--n_pass', default=10, type=int)
parser.add_argument('--q_pass', default=10, type=int)
parser.add_argument('--k_pass', default=10, type=int)
parser.add_argument('--bond_q_pass', default=10, type=int)
parser.add_argument('--bond_k_pass', default=10, type=int)
parser.add_argument('--freq_type', default='tkauto', type=str, choices=['ori', 'tkdft', 'tkauto'])
parser.add_argument('--step', default=0, type=int)
parser.add_argument('--lr_for_selector', default=1e-3, type=float)
parser.add_argument('--probe_layers', default='-1', type=str)
parser.add_argument('--result_file', default='roberta_base.json', type = str)
parser.add_argument('--chpt_dir', default='save_models', type=str)
parser.add_argument('--max_len', default=120, type=int)
parser.add_argument('--sparse_hold', default=0.5, type=float)
parser.add_argument('--pca_k', default=2, type=int)

parser.add_argument('--gnn_or_mlp', default='gnn', type=str, choices=['mlp', 'gnn'])

args = parser.parse_args()
set_seed(args.seed)

if args.dataset == 'ec':
    args.data_dir = r"/nfsfile/niuhao/project/html_www2020/raw_data/ReleasedDataset_mp3_aspect/"
elif args.dataset == '1516':
    args.data_dir = r"/nfsfile/niuhao/project/html_www2020/MAECdata/MAEC_Dataset_aspect/"


fitlog.add_hyper_in_file(__file__)
fitlog.add_hyper(args)


print(args)
args.result_save_path = '/nfsfile/niuhao/project/aspect_process_for_thesis/FLT/Train/FT_logs' + args.result_file
with open(args.result_save_path, 'a') as fp:
    json.dump(vars(args), fp, indent=4)
#######hyper
n_epochs = 60    #20    60
pool = "max"
smooth_eps = 0.0
dropout = 0.5
#######hyper


model_type = args.model_name.split("-")[0]
if model_type == "bert":
    mask = "[UNK]"
elif model_type == "roberta":
    mask = "<mask>"
elif model_type == "xlnet":
    mask = "<mask>"
elif model_type == "xlmroberta":
    mask = "<mask>"


@cache_results(
    f"{root_fp}/caches/data_{args.dataset}_{mask}_{args.model_name}.pkl",
    _refresh=False,
)
def get_data():
    data_bundle, allembeds = DataPipe(model_name=args.model_name, mask=mask).process_from_file(
        os.path.join(args.data_dir)
    )
    return data_bundle, allembeds


data_bundle, allembeds = get_data()

print(data_bundle)

if args.model_name.split("-")[0] in ("bert", "roberta", "xlnet", "xlmroberta"):
    model_type, model_name = (
        args.model_name[: args.model_name.index("-")],
        args.model_name[args.model_name.index("-") + 1 :],
    )

if model_type == "roberta":
    if model_name == 'en-large':
        embed = RobertaWordPieceEncoder('/nfsfile/niuhao/.fastNLP/embedding/roberta-large/', requires_grad=True)
    elif model_name == 'en':
        embed = RobertaWordPieceEncoder('/nfsfile/niuhao/.fastNLP/embedding/roberta-base/', requires_grad=True)
    # embed = RobertaWordPieceEncoder(model_dir_or_name=model_name, requires_grad=True)
elif model_type == "bert":
    embed = BertWordPieceEncoder('/nfsfile/niuhao/.fastNLP/embedding/bert-base-uncased', requires_grad=True)
elif model_type == "xlnet":
    embed = XLNetModel.from_pretrained(pretrained_model_name_or_path=model_name)
elif model_type == "xlmroberta":
    embed = XLMRobertaModel.from_pretrained(pretrained_model_name_or_path=model_name)

# for n, p in embed.named_parameters():
#     p.requires_grad = False

if args.gnn_or_mlp == 'mlp':
    model = MlpModel(
        args,
        embed,
        dropout=dropout,
        num_classes=len(data_bundle.get_vocab("target")) - 1,
        pool=pool,
    )
else:
    model = AspectModel(
        args,
        embed,
        dropout=dropout,
        num_classes=len(data_bundle.get_vocab("target")) - 1,
        pool=pool,
    )

ckpt_path_dir = r"/nfsfile/niuhao/project/aspect_process_for_thesis/FLT/Train/tkauto_best_rbtbase/best_AspectModel_acc_2024-06-15-06-21-20-440968"
model.load_state_dict(torch.load(ckpt_path_dir).state_dict())
model = model.cuda()

if args.multi_gpus:
    model = nn.DataParallel(model, device_ids=[0,1])

tokenizer = BertTokenizer.from_pretrained('/nfsfile/niuhao/.fastNLP/embedding/bert-base-uncased')

for name, ds in tqdm(data_bundle.iter_datasets()):
    name_dir = allembeds[name]
    tr_data = DataSetIter(
        data_bundle.get_dataset(name),
        num_workers=1,
    )

    for batch in tr_data:
        sent_key = ' '.join(batch[0]['raw_tokens'].squeeze().tolist())
        _, aspect_embed = model(batch[0]["tokens"].cuda(), batch[0]["aspect_mask"].cuda())
        name_dir[sent_key] += [aspect_embed.cpu().detach()]

    if args.dataset == 'ec':
        f = open('/nfsfile/niuhao/project/html_www2020/raw_data/ReleasedDataset_mp3_aspect/'+ name + '/ec_flt_pkls.pkl', 'wb')
    elif args.dataset == '1516':
        f = open('/nfsfile/niuhao/project/html_www2020/MAECdata/MAEC_Dataset_aspect/'+ name + '/maec_flt_pkls.pkl', 'wb')
    
    pickle.dump(name_dir, f)


if args.save_embed:
    os.makedirs(f"{root_fp}/" +  args.chpt_dir, exist_ok=True)
    folder = f"{root_fp}/{args.chpt_dir}/{model_type}-{args.dataset}-FT"
    count = 0
    for fn in os.listdir(f"{root_fp}/{args.chpt_dir}"):
        if fn.startswith(folder.split("/")[-1]):
            count += 1
    folder = folder + str(count)
    fitlog.add_other(count, name="count")
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        if model_type  in ('bert', 'roberta'):
            embed.save(folder)
        else:
            embed.save_pretrained(folder)
