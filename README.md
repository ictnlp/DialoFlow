# Conversations Are Not Flat: Modeling the Dynamic Information Flow across Dialogue Utterances
This repository contains the code and pre-trained models for our ACL 2021 paper Conversations Are Not Flat: Modeling the Dynamic Information Flow across Dialogue Utterances [pdf](https://arxiv.org/abs/2106.02227). 

**************************** **Updates** ****************************

The Chinese version is comming soon!

- 6/30: We released the code and pre-trained model (English version) of **DialoFlow**. 

* 5/10: We released the code and pre-trained model of **Flow Score**. Try to use it!




## Overview

We propose the **DialoFlow**, a new paradigm to construct the dynamic information flow in the dialogue history by addressing the semantic influence brought about by each utterance. Besides, we design an automatic reference-free evaluation metric **Flow Score** based on the pre-trained DialoFlow for interactive dialogue quality evaluation.

![Overview of DialoFlow](figure/model.png)



## DialoFlow

### Requirements

torch==1.7.0

transformers==3.0.2

apex

### Pre-trained models

DialoFlow is pre-trained on the Reddit dataset based on the GPT-2. 

For more details about the dataset, please refer to [DialoGPT](https://github.com/microsoft/DialoGPT).

We release three pre-trained models: [DialoFlow_base](https://drive.google.com/drive/folders/1yK__2CdD_4Ca3d02HkAph6ndkkgVR3YU?usp=sharing), [DialoFlow_medium](https://drive.google.com/drive/folders/12acVZVXu7dmeB-jBocEJSrMNytuai0CU?usp=sharing), and [DialoFlow_large](https://drive.google.com/drive/folders/11a2WZezOCvV652QSTYgZkAZ1nezXrhhi?usp=sharing).

Please download the pre-trained models under the path `models/`.

The fine-tuning models on the BST dataset and the Chinese version will be public soon.



### Dialogue Generation

We provide the code for dialogue generation using the pre-trained DialoFlow model. 

The script `generate.py` contains two decoding methods: beam search and nucleus sampling.

You can modify the code for your own data and task.



### Fine-tuning

We fine-tuned the pre-trained model on the DailyDialog dataset. 

```shell
cd dailydialog
bash fine-tune.sh
```



## Flow Score

**Flow Score** is an automatic reference-free evaluation metric for interactive dialogue evaluation based on the pre-trained DialoFlow. **Flow Score** can be found [here](https://github.com/ictnlp/DialoFlow/tree/main/FlowScore).



## Citation

Please cite our paper if you use DialoFlow in your work.

```bibtex
@inproceedings{li2021dialoflow,
   title={Conversations are not Flat: Modeling the Dynamic Information Flow across Dialogue Utterances},
   author={Li, Zekang and Zhang, Jinchao and Fei, Zhengcong and Feng, Yang and Zhou, Jie},
   booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics},
   year={2021}
}
```
