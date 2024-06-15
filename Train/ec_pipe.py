import json
import os
import warnings
import pickle
from tqdm import tqdm

warnings.filterwarnings("ignore")

from fastNLP import DataSet, Instance, Vocabulary
from fastNLP.io import DataBundle, Loader, Pipe
from fastNLP.modules.tokenizer import BertTokenizer, RobertaTokenizer
from transformers import XLMRobertaTokenizer, XLNetTokenizer


class DataPipe(Pipe):
    def __init__(self, model_name="en", mask="<mask>"):
        super().__init__()
        if model_name.split("-")[0] in ("bert", "roberta", "xlnet", "xlmroberta"):
            model_type, model_name = (
                model_name[: model_name.index("-")],
                model_name[model_name.index("-") + 1 :],
            )
        else:
            raise ValueError("Invalid model name")
        if model_type == "roberta":
            if model_name == 'en-large':
                self.tokenizer = RobertaTokenizer.from_pretrained(
                    '/nfsfile/niuhao/.fastNLP/embedding/roberta-large/'
                )
            elif model_name == 'en':
                self.tokenizer = RobertaTokenizer.from_pretrained(
                    '/nfsfile/niuhao/.fastNLP/embedding/roberta-base/'
                )
        # if model_type == "roberta":
        #     self.tokenizer = RobertaTokenizer.from_pretrained(
        #         model_dir_or_name=model_name
        #     )
        elif model_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained('/nfsfile/niuhao/.fastNLP/embedding/bert-base-uncased')
        elif model_type == "xlnet":
            self.tokenizer = XLNetTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name
            )
        elif model_type == "xlmroberta":
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name
            )

        self.mask = mask

    def process(self, data_bundle: DataBundle) -> DataBundle:
        new_bundle = DataBundle()
        if isinstance(self.tokenizer, BertTokenizer):
            cls = "[CLS]"
            sep = "[SEP]"
        else:
            cls = self.tokenizer.cls_token
            sep = self.tokenizer.sep_token

        allembeds = {}
        for name, ds in tqdm(data_bundle.iter_datasets()):
            new_ds = DataSet()
            allembeds[name] = {}
            for ins in ds:
                tokens = ins["tokens"]
                if not isinstance(self.tokenizer, XLNetTokenizer):
                    tokens.insert(0, cls)
                    tokens.append(sep)
                    shift = 1
                else:
                    tokens.append(sep)
                    tokens.append(cls)
                    shift = 0

                allembeds[name][' '.join(tokens)] = []

                for aspect in ins["aspects"]:
                    target = aspect["polarity"]
                    start = aspect["from"] + shift
                    end = aspect["to"] + shift
                    aspect_mask = [0] * len(tokens)
                    for i in range(start, end):
                        aspect_mask[i] = 1
                    pieces = []
                    piece_masks = []
                    raw_words = tokens[shift:-1]
                    raw_words.insert(start - 1, "[[")
                    raw_words.insert(end, "]]")
                    for mask, token in zip(aspect_mask, tokens):
                        bpes = self.tokenizer.convert_tokens_to_ids(
                            self.tokenizer.tokenize(token)
                        )
                        pieces.extend(bpes)
                        piece_masks.extend([mask] * (len(bpes)))
                    new_ins = Instance(
                        raw_tokens = tokens,
                        tokens=pieces,
                        target=target,
                        aspect_mask=piece_masks,
                        raw_words=" ".join(raw_words),
                    )
                    new_ds.append(new_ins)
            new_bundle.set_dataset(new_ds, name)
            # if new_bundle.num_dataset > 10:
            #     break

        target_vocab = Vocabulary(padding=None, unknown=None)
        target_vocab.add_word_lst(["neutral", "positive", "negative", "smooth"])
        # target_vocab.index_dataset(*new_bundle.datasets.values(), field_name="target")

        new_bundle.set_target("target")
        new_bundle.set_input("tokens", "aspect_mask", "raw_words", "raw_tokens")
        new_bundle.apply_field(
            lambda x: len(x), field_name="tokens", new_field_name="seq_len"
        )

        # new_bundle.set_vocab(vocab, 'tokens')
        if hasattr(self.tokenizer, "pad_token_id"):
            new_bundle.set_pad_val("tokens", self.tokenizer.pad_token_id)
        else:
            new_bundle.set_pad_val("tokens", self.tokenizer.pad_index)
        new_bundle.set_vocab(target_vocab, "target")

        return new_bundle, allembeds

    def process_from_file(self, paths) -> DataBundle:
        data_bundle = DataLoader().load(paths)
        new_bundle, allembeds = self.process(data_bundle)

        return new_bundle, allembeds

class DataLoader(Loader):
    def __init__(self):
        super().__init__()

    def load(self, paths):
        """
        输出的DataSet包含以下的field
        tokens                  pos                   dep                                    aspects
        ["The", "bread", ...]   ["DET", "NOUN",...]   [["dep", 2, 1], ["nsubj", 4, 2], ...]  [{"term": ["bread"], "polarity": "positive", "from": 1, "to": 2}]
        其中dep中["dep", 2, 1]指当前这个word的head是2（0是root，这里2就是bread），"dep"是依赖关系为dep

        :param paths:
        :return:
        """
        data_bundle = DataBundle()
        
        for name in os.listdir(paths):
            fp = os.path.join(paths, name, 'text_aspect.pkl')
            with open(fp, "rb") as f:
                data = pickle.load(f)
            data = data['sent_aspect']
            ds = DataSet()
            for ins in data:
                tokens = ins["sent_spy"]
                pos = ins["pos"]
                dep = [[dtk, h, t]for t, (dtk, h) in enumerate(zip(ins["dep"], ins['head_ids']))]
                if 'aspect' in ins.keys():
                    aspects = [
                        {
                            'term': ins["aspect"].split(' '),
                            'polarity': "positive",
                            'from': ins["aspect_start"],
                            'to': ins['aspect_end']
                        }
                    ]
                else:
                    aspects = []
                ins = Instance(tokens=tokens, pos=pos, dep=dep, aspects=aspects)
                ds.append(ins)
            data_bundle.set_dataset(ds, name=name)
        return data_bundle
