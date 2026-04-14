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
"""
Validation-only entry point using the API model path.
No Ray cluster, no training loop — just config + tokenizer + validation.
"""
from .agent_ray_trainer_retro_noback import ValidationPipeline

from agent_r1.tool import ToolEnvRetroNoBack
from agent_r1.tool.tools import _default_tools

import hydra

from verl import DataProto
from .reward_score import _default_compute_score_format, _default_compute_score_answer, _default_compute_score_format_answer, _default_compute_all_score
import torch

class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

    def __call__(self, data: DataProto, env: 'ToolEnvRetroNoBack'):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        end_lst = []
        answer_lst = []
        format_lst = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            env_item = env[i]

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()

            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']

            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

            valid_response_ids = response_ids[:valid_response_length].long()

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))

            sequences_str = self.tokenizer.decode(sequences, skip_special_tokens=False)
            pad_token_id = self.tokenizer.pad_token_id
            sequences_str = sequences_str.split(self.tokenizer.decode([pad_token_id]))[0]

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']

            score, end_score, answer_score, format_score = _default_compute_all_score(data_source=data_source, solution_str=sequences_str, ground_truth=ground_truth, env=env_item)

            end_lst.append(end_score)
            answer_lst.append(answer_score)
            format_lst.append(format_score)

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt+response]", sequences_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        return reward_tensor, end_lst, answer_lst, format_lst

@hydra.main(config_path='config', config_name='agent_trainer', version_base=None)
def main(config):
    run_agent(config)


def run_agent(config) -> None:
    from verl.utils.fs import copy_to_local
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer, hf_processor
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

    tools = _default_tools(config.tool.env)
    env = ToolEnvRetroNoBack(tools=tools, max_turns=config.tool.max_turns, maxstep=config.tool.maxstep, topk=config.tool.topk, shuffle=config.tool.shuffle, force_noloop=config.tool.force_noloop)

    num_examine = 2

    pipeline = ValidationPipeline(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        val_reward_fn=RewardManager(tokenizer=tokenizer, num_examine=num_examine),
        env=env,
    )
    pipeline.run()


if __name__ == '__main__':
    main()
