## Flow Score

**Flow Score** is an automatic reference-free evaluation metric for interactive dialogue evaluation based on the pre-trained DialoFlow.

As the **DialoFlow** is trained on human-human dialogues, the context flow scheme can be regarded as the general expectation of the dialogue development. Therefore, the closer gap between the semantic influence brought by the chatbot’s utterance and the expectation means the more human-likeness. 

## How to use?

```python
from flow_score import *
MODEL_PATH = "models/DialoFlow_large.bin"
FLOW_SCORE = FlowScore(MODEL_PATH)
dialogues = ["hello", "Hi there. tell me about yourself.", "Well I'm a college student who loves learning about the world around me!"]
flow_score = FLOW_SCORE.score(dialogues)
```

Please download the pre-trained model from [here](https://drive.google.com/file/d/19-v96TMevn22h54POJUHkYsSLTOrYHcr/view?usp=sharing).

## Requirements

torch==1.7.0

transformers==3.0.2

## Data

We use the dialogues from the Interactive Evaluation of Dialog Track @ DSTC9. The dialogues are from 11 different chatbots. 

Please cite DSTC9 paper if you use the data. 



```bibtex
@article{gunasekara2020overview,
  title={Overview of the Ninth Dialog System Technology Challenge: DSTC9},
  author={Gunasekara, Chulaka and Kim, Seokhwan and D'Haro, Luis Fernando and Rastogi, Abhinav and Chen, Yun-Nung and Eric, Mihail and Hedayatnia, Behnam and Gopalakrishnan, Karthik and Liu, Yang and Huang, Chao-Wei and others},
  journal={Proceedings of the 9th Dialog System Technology Challenge Workshop in AAAI2021},
  url = {https://arxiv.org/abs/2011.06486},
  year={2021}
}
```



## Citation

Please cite our paper if you use **Flow Score** in your work:

```bibtex
@inproceedings{li2021dialoflow,
   title={Conversations are not Flat: Modeling the Intrinsic Information Flow between Dialogue Utterances},
   author={Li, Zekang and Zhang, Jinchao and Fei, Zhengcong and Feng, Yang and Zhou, Jie},
   booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics},
   year={2021}
}
```
