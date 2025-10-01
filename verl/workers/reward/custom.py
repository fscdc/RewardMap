# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from transformers import PreTrainedTokenizer
import json

from verl import DataProto
from verl.utils.reward_score import math_compute_score, r1v_compute_score, seg_compute_score, seg_strict_compute_score, vision_reasoner_compute_score, reason_map_compute_score


class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, num_examine: int, compute_score: str):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        if compute_score == "math":
            self.compute_score = math_compute_score
        elif compute_score == "r1v":
            self.compute_score = r1v_compute_score
        elif compute_score == "seg":
            self.compute_score = seg_compute_score
        elif compute_score == "seg_strict":
            self.compute_score = seg_strict_compute_score
        elif compute_score == "vision_reasoner":
            self.compute_score = vision_reasoner_compute_score
        elif compute_score == "reason_map":
            self.compute_score = reason_map_compute_score
        else:
            raise NotImplementedError()

    def __call__(self, data: DataProto) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print = 0

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)


            # items from ReasonMap
            ground_truth = data_item.non_tensor_batch["answer"] # int, 1 means yes for 'TorF' questions
            type_reasonmap = data_item.non_tensor_batch["type"] # str
            difficulty_city = data_item.non_tensor_batch["difficulty_city"] # str # [easy, middle, hard]
            city_line_count = data_item.non_tensor_batch["city_line_count"] # int 
            city_transfer_count = data_item.non_tensor_batch["city_transfer_count"] # int
            question_transfer_count = data_item.non_tensor_batch["question_transfer_count"] # int
            country = data_item.non_tensor_batch["country"] # str
            city = data_item.non_tensor_batch["city"] # str
            station1 = data_item.non_tensor_batch["station_1"] # str
            station2 = data_item.non_tensor_batch["station_2"] # str
            json_path = data_item.non_tensor_batch["json"] # str, path to json file
            
            meta_metro_data = None
            with open(json_path, 'r', encoding='utf-8') as f:
                meta_metro_data = json.load(f)

            metro_data = {}
            for route_name, stations in meta_metro_data.items():
                metro_data[route_name] = [
                    s.replace(" (Transfer Station)", "").replace(" (Branch-starting Station)", "").replace("（换乘站）", "").replace("（支线起始站）", "")
                    for s in stations
                ]

            # @sicheng, kaiwen: two modes: difficulty_aware / baseline
            score = self.compute_score(
                response_str, 
                ground_truth, 
                station1, 
                station2, 
                metro_data, 
                "difficulty_aware", 
                type_reasonmap,
                difficulty_city,
                question_transfer_count, 
            )
            reward_tensor[i, valid_response_length - 1] = score


            if already_print < self.num_examine:
                already_print += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        return reward_tensor
