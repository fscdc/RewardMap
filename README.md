<div align="center">
      <h2><b> RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning </b></h2>
</div>

<div align="center">

![](https://img.shields.io/github/stars/fscdc/RewardMap?color=yellow)
![](https://img.shields.io/github/forks/fscdc/RewardMap?color=lightblue)
![](https://img.shields.io/github/last-commit/fscdc/RewardMap?color=green)
![](https://img.shields.io/badge/PRs-Welcome-blue)
<a href="" target="_blank"><img src="https://img.shields.io/badge/arXiv-TODO-009688.svg" alt="arXiv"></a>

</div>

---

>🙋 Please let us know if you find out a mistake or have any suggestions!
>
>🌟 If you find this resource helpful, please consider to star this repository and cite our [research](#citation)!

<p align="center">
<img src="assets/rewardmap.svg" width = "95%" alt="" align=center />
</p>

## Updates

- 2025-10-01: 🚀 We released `RewardMap` and the corresponding [ReasonMap-Plus](https://huggingface.co/datasets/FSCCS/ReasonMap-Plus)!

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

If you find this paper useful in your research, please consider citing our paper:

```bibtex
@article{feng2025rewardmap,
  TODO
}
```
