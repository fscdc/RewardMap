<div align="center">
      <h2><b> [ICLR 2026] RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning </b></h2>
</div>

<div align="center">

![](https://img.shields.io/github/stars/fscdc/RewardMap?color=yellow)
![](https://img.shields.io/github/forks/fscdc/RewardMap?color=lightblue)
![](https://img.shields.io/github/last-commit/fscdc/RewardMap?color=green)
![](https://img.shields.io/badge/PRs-Welcome-blue)
<a href="https://arxiv.org/abs/2510.02240" target="_blank"><img src="https://img.shields.io/badge/arXiv-2510.02240-009688.svg" alt="arXiv"></a>
[![Dataset](https://img.shields.io/badge/🤗%20Huggingface-Dataset-yellow)](https://huggingface.co/collections/FSCCS/reasonmap)

</div>

<div align="center">

**[<a href="https://huggingface.co/papers/2510.02240">HuggingFace Daily Paper</a>]** **[<a href="https://x.com/si_feng32704/status/1973968606468997410">Twitter</a>]** **[<a href="https://mp.weixin.qq.com/s/jTnrxfZ7Secq1-ZO1mDMyg">机器之心</a>]**

</div>

This repository is for our paper:

> **[RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning](https://arxiv.org/abs/2510.02240)** \
> [Sicheng Feng](https://fscdc.github.io/)<sup>1,^</sup>, [Kaiwen Tuo](https://cfintech.github.io/)<sup>1,2,^</sup>, [Song Wang](https://songw-zju.github.io/)<sup>3</sup>, [Lingdong Kong](https://ldkong.com/)<sup>4</sup>, [Jianke Zhu](https://person.zju.edu.cn/en/jkzhu)<sup>3</sup>, [Huan Wang]()<sup>1,*</sup> \
> <sup>1</sup>Westlake University, Hangzhou, China \
> <sup>2</sup>Tongji University, Shanghai, China \
> <sup>3</sup>Zhejiang University, Hangzhou, China \
> <sup>4</sup>National University of Singapore, Singapore \
> <sup>^</sup>Equal contribution, <sup>∗</sup>Corresponding author: wanghuan@westlake.edu.cn

---

>🙋 Please let us know if you find a mistake or have any suggestions!
>
>🌟 If you find this resource helpful, please consider to star this repository and cite our [research](#citation)!

<p align="center">
<img src="assets/rewardmap.svg" width = "95%" alt="" align=center />
</p>

## Updates

- 2026-01-26: 📢 Our paper was accepted by ICLR 2026! Thanks to all contributors!
- 2025-10-27: 📢 You can use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to evaluate ReasonMap-Plus! Free to ask when facing a problem!
- 2025-10-03: 📢 Our paper "RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning" is now available on [arXiv](https://arxiv.org/abs/2510.02240)!
- 2025-10-01: 🚀 Based on [ReasonMap](https://arxiv.org/abs/2505.18675), we released `RewardMap` and the corresponding [ReasonMap-Plus](https://huggingface.co/datasets/FSCCS/ReasonMap-Plus)!

## Usage

### 1. Install dependencies

If you face any issues with the installation, please feel free to open an issue. We will try our best to help you.

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

<p align="center">
<img src="assets/overview_dataset.svg" width = "95%" alt="" align=center />
</p>

You can download [ReasonMap-Plus](https://huggingface.co/datasets/FSCCS/ReasonMap-Plus) for evaluation and [ReasonMap-Train](https://huggingface.co/datasets/FSCCS/ReasonMap-Train) for Rewardap Training from HuggingFace or by running the following command:

```bash
python utils/download_dataset.py
```

Then, put the data under the folder `data`.


### 3. Training

You can train the model by running the following command:

```bash
# RewardMap training
bash scripts/reward_map.sh
```

Then, you can merge the trained model by running:

```bash
# merge trained model
bash scripts/merge_model.sh
```

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to conduct SFT training. Please first put the file `sft.yaml` under the folder `examples/train_full` of `LLaMA-Factory` repo and prepare the datasets by running the following command:

```bash
python utils/prepare_data_for_sft.py --dataset_dir path/to/your_data
```

Your data will be transferred into the format like:

```json
  {
    "conversations": [
      {
        "from": "human",
        "value": "<image> Please solve the multiple choice problem and put your answer (one of ABCD) in one \"\\boxed{}\". According to the subway map, how many intermediate stops are there between Danube Station and lbn Battuta Station (except for this two stops)? \nA) 8 \nB) 1 \nC) 25 \nD) 12 \n"
      },
      {
        "from": "gpt",
        "value": "B"
      }
    ],
    "images": [
      "./maps/united_arab_emirates/dubai.png"
    ]
  },
```
Then, add the data information in the file `LLaMA-Factory/data/dataset_info.json`:

```json
  "reasonmap_plus": {
    "file_name": "reason_map_plus.json",
    "formatting": "sharegpt",
    "ranking": false,
    "columns": {
      "messages": "conversations",
      "images": "images"
    }
  }
```

Then run the following command under the `LLaMA-Factory` repo:

```bash
# SFT training
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/reason-map-plus.yaml
```

### 4. Evaluation

You can evaluate the model performance on `ReasonMap` or `ReasonMap-Plus` by following the guideline in [ReasonMap](https://github.com/fscdc/ReasonMap).


We use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to evaluate our models on other benchmarks, to conduct evaluation, you should first add the model information in `VLMEvalKit/vlmeval/config.py`:

```python
"your-model-name": partial(
    Qwen2VLChat,
    model_path="path/to/your_model",
    min_pixels=1280 * 28 * 28,
    max_pixels=16384 * 28 * 28,
    use_custom_prompt=False,
),
```

Then run the following command under the `VLMEvalKit` repo:

```bash
# evaluate on other benchmarks
bash script/eval_other_benchmarks.sh
```

## Acknowledgement

This source code is derived from the PyTorch reimplementation of [Seg-Zero](https://github.com/dvlab-research/Seg-Zero).

## Citation

If you find this paper useful in your research, please consider citing our papers:

```bibtex
@article{feng2025can,
  title={Can MLLMs Guide Me Home? A Benchmark Study on Fine-Grained Visual Reasoning from Transit Maps},
  author={Feng, Sicheng and Wang, Song and Ouyang, Shuyi and Kong, Lingdong and Song, Zikai and Zhu, Jianke and Wang, Huan and Wang, Xinchao},
  journal={arXiv preprint arXiv:2505.18675},
  year={2025}
}

@article{feng2025rewardmap,
  title={RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning},
  author={Feng, Sicheng and Tuo, Kaiwen and Wang, Song and Kong, Lingdong and Zhu, Jianke and Wang, Huan},
  journal={arXiv preprint arXiv:2510.02240},
  year={2025}
}
```
