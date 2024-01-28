# Adaptive Structure Induction for Aspect-based Sentiment Analysis with Spectral Perspective

Official implementation of ['Adaptive Structure Induction for Aspect-based Sentiment Analysis with Spectral Perspective'](https://aclanthology.org/2023.findings-emnlp.79/).

The paper has been accepted by **EMNLP-Findings 2023** 🔥.


## Abstract
Financial volatility prediction is vital for characterizing a company’s risk profile. Transcripts of companies’ earnings calls serve as valuable, yet unstructured, data sources to be utilized to access companies’ performance and risk profiles. Despite their importance, current works ignore the role of financial metrics knowledge (such as EBIT, EPS, and ROI) in transcripts, which is crucial for understanding companies’ performance, and little consideration is given to integrating text and price information. In this work, we statistically analyze common financial metrics and create a special dataset centered on these metrics. Then, we propose a knowledge-enhanced financial volatility prediction method (KeFVP) to inject knowledge of financial metrics into text comprehension by knowledge-enhanced adaptive pre-training (KePt) and effectively integrating text and price information by introducing a conditional time series prediction module. Extensive experiments are conducted on three realworld public datasets, and the results indicate that KeFVP is effective and outperforms all the state-of-the-art methods.

<div align="center">
  <img src="figs/flt.png"/>
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