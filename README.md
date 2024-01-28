# Adaptive Structure Induction for Aspect-based Sentiment Analysis with Spectral Perspective

Official implementation of ['Adaptive Structure Induction for Aspect-based Sentiment Analysis with Spectral Perspective'](https://aclanthology.org/2023.findings-emnlp.79/).

The paper has been accepted by **EMNLP-Findings 2023** 🔥.


## Abstract
Recently, incorporating structure information (e.g. dependency syntactic tree) can enhance the performance of aspect-based sentiment analysis (ABSA). However, this structure information is obtained from off-the-shelf parsers, which is often sub-optimal and cumbersome. Thus, automatically learning adaptive structures is conducive to solving this problem. In this work, we concentrate on structure induction from pre-trained language models (PLMs) and throw the structure induction into a spectrum perspective to explore the impact of scale information in language representation on structure induction ability. Concretely, the main architecture of our model is composed of commonly used PLMs (e.g., RoBERTa, etc.), anda simple yet effective graph structure learning (GSL) module (graph learner + GNNs). Subsequently, we plug in Frequency Filters with different bands after the PLMs to produce filtered language representations and feed them into the GSL module to induce latent structures. We conduct extensive experiments on three public benchmarks for ABSA. The results and further analyses demonstrate that introducing this spectral approach can shorten Aspects-sentiment Distance (AsD) and be beneficial to structure induction. Even based on such a simple framework, the effects on three datasets can reach SOTA (state-of-the-art) or near SOTA performance. Additionally, our exploration also has the potential to be generalized to other tasks or to bring inspiration to other similar domains.

<div align="center">
  <img src="figs/flt.png", width='90%' />
</div>


## Requirements

### Installation
Create a conda environment and install dependencies:
```bash
git clone git@github.com:hankniu01/FLT.git
cd flt

conda create -n flt python=3.8
conda activate flt

pip install -r requirements.txt

```

## Get Started

### Running FLT

```bash
cd Train
bash ./scripts/flt.sh
bash ./scripts/flt_rbtlarge.sh
```
### Running AFS
```bash
cd Train
bash ./scripts/afs.sh
```

## Acknowledgement
This repo benefits from [RoBERTaABSA](https://github.com/ROGERDJQ/RoBERTaABSA). Thanks for their wonderful works.


## Citation
```bash
@inproceedings{niu-etal-2023-adaptive,
    title = "Adaptive Structure Induction for Aspect-based Sentiment Analysis with Spectral Perspective",
    author = "Niu, Hao  and
      Xiong, Yun  and
      Wang, Xiaosu  and
      Yu, Wenjing  and
      Zhang, Yao  and
      Guo, Zhonglei",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.79",
    doi = "10.18653/v1/2023.findings-emnlp.79",
    pages = "1113--1126"
}
```