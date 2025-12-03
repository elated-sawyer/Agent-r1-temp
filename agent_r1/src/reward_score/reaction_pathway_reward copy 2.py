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

import re
import string
import json

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0.0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1.0
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0.0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1.0
            break
    return score


def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return None

def compute_score_format(solution_str, env):
    """The scoring function for format reward.

    Args:
        solution_str: the solution text
    
    """
    if solution_str is None or env is None:
        return 0.0, 0.0
    
    try:
        for key, value in env.unsolved_dict.items():
            if not value:
                end = key
                break
        else:
            end = None
        
        if end is None:
            return 0.0, 0.0
        
        end_reward = 1.0
        steps = env.steps_taken
        actions_valid = env._actions_valid
        actions_effective = env._actions_effective

        format_reward = 0.0
        for i in range(len(actions_valid)):
            if not actions_valid[i] or not actions_effective[i]:
                format_reward -= 0.01
        if steps > 40:
            format_reward -= 0.2
        if steps > 60:
            format_reward -= 0.2
        if steps > 80:
            format_reward -= 0.2
            format_reward -= (steps-80)*0.01
        
        format_reward = max(format_reward, -0.8)

        '''
        end_reward = 0.0
        if end is not None:
            end_reward = 1.0
            return end_reward, 0.0
        
        # Perfect format match for the new structure
        # First <|im_start|>assistant should have <think> and possibly <tool_call>
        # Then <|im_start|>tool with <tool_response> (can repeat with assistant/tool pairs)
        # Final <|im_start|>assistant with the answer and <|im_end|>
        
        # Check for basic structure with <|im_start|>assistant and <|im_end|> tags
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)

        format_reward = 0.0
        
        # If no blocks found, return 0
        if not assistant_blocks:
            return end_reward, format_reward
        
        env_rewards = env.rewards
        # Perfect format requires at least one assistant block and matching tool blocks if tool calls exist
        # Check first assistant block contains <think> tags
        for i, assistant_block in enumerate(assistant_blocks[:-1]):
            if assistant_block.count('<think>') == 1 and assistant_block.count('</think>') == 1 and assistant_block.count('<tool_call>') == 1 and assistant_block.count('</tool_call>') == 1:
                think_match = re.search(r'^<think>(.*?)</think>\n<tool_call>(.*?)</tool_call>$', assistant_block, re.DOTALL)
                # soft_think_match = re.search(r'<think>(.*?)</think>(.*?)<tool_call>(.*?)</tool_call>', assistant_block, re.DOTALL)
                if think_match:
                    # format_reward += 0.05 * (0.95 ** i)
                    if i < len(env_rewards) and env_rewards[i] > 0:
                        format_reward += 0.05
                        if format_reward >= 0.7:
                            break
        '''

        # Check the last assistant block contains <answer> tags
        # if assistant_blocks:  # 确保有至少一个assistant块
        #     last_assistant_block = assistant_blocks[-1]
        #     think_answer_match = re.search(r'^<think>(.*?)</think>\n<answer>(.*?)</answer>$', last_assistant_block, re.DOTALL)
        #     if think_answer_match:
        #         format_reward += 0.5
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format: {e}")
        return 0.0, 0.0
    
    return end_reward, format_reward


def compute_score_answer(solution_str, ground_truth, env):
    """The scoring function for exact match (EM) with format reward.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    
    Returns:
        float: Total reward score (format reward + answer reward)
    """
    if solution_str is None or env is None:
        return 0.0
    
    try:
        for key, value in env.unsolved_dict.items():
            if not value:
                end = key
                break
        else:
            end = None
        
        if end is None:
            return 0.0
        
        # Extract answer from <answer> tags
        # assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        # solution_str = assistant_blocks[-1]
        # answer = extract_solution(solution_str)

        
        # if isinstance(ground_truth, list):
        #     ground_truth = ground_truth[0]
        # ground_truth_length = len(json.loads(ground_truth))

        step = env.step_dict[end]
        answer_reward = 1.2-step*0.05
        # answer_reward = ground_truth_length*0.05 + (ground_truth_length-step) * 0.1 + 0.1
        answer_reward = max(answer_reward, 0.2)

        '''
        answer_reward = 0.6
        if step > 0:
            if step <= ground_truth_length:
                answer_reward += (ground_truth_length-step)/ground_truth_length/2
            else:
                answer_reward -= min((step-ground_truth_length)/ground_truth_length/2, 0.4)
            # answer_reward += 0.5/step
        '''
        
        # first_end = 0
        # user_blocks = re.findall(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        # for i, user_block in enumerate(user_blocks):
        #     if 'You have successfully found a pathway to synthesize the target molecule' in user_block:
        #         first_end = i
        #         break
        
        # if first_end > 0:
        #     answer_reward += 0.5/first_end


        # if answer is not None:
        #     # Check for exact match within <answer>
        #     # if em_check(answer, ground_truth):
        #     #     answer_reward = 1.0
        #     # # Check for substring match within <answer>
        #     # elif subem_check(answer, ground_truth):
        #     #     answer_reward = 0.5
        #     if subem_check(answer, ground_truth):
        #         answer_reward = 1.0
        
        # If no match found within <answer>, check entire solution for substring match
        # if answer_reward == 0.0:
        #     if subem_check(solution_str, ground_truth):
        #         answer_reward = 0.2
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_answer: {e}")
        return 0.0
    
    return answer_reward

def compute_score_format_answer(solution_str, ground_truth):
    """The scoring function for format reward.

    Args:
        solution_str: the solution text
    
    """
    if solution_str is None or ground_truth is None:
        return 0.0

    try:
        format_reward = compute_score_format(solution_str)
        answer_reward = compute_score_answer(solution_str, ground_truth)

        format_reward = min(format_reward, 1.0)
        if format_reward == 1.0:
            return -1.0 + format_reward + answer_reward
        else:
            return -1.0 + format_reward
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format_answer: {e}")
        return 0.0

def compute_score_em(solution_str, ground_truth):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    
    """
    if solution_str is None or ground_truth is None:
        return 0.0
    
    try:
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        solution_str = assistant_blocks[-1]
        answer = extract_solution(solution_str)
        if answer is None:
            return 0.0
        return float(subem_check(answer, ground_truth))
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_em: {e}")
        return 0.0


def compute_all_score(solution_str, ground_truth, env):
    if solution_str is None:
        end_reward = 0.0
        format_reward = 0.0
        answer_score = 0.0
        score = -1.0
        return score, end_reward, answer_score, format_reward
    
    try:
        end_reward, format_reward = compute_score_format(solution_str, env)
        answer_reward = compute_score_answer(solution_str, ground_truth, env)

        format_reward = min(format_reward + end_reward, 1.0)
        score = -1.0 + format_reward + answer_reward
        return score, end_reward, answer_reward, format_reward
        # if format_reward == 1.0:
        #     return -1.0 + format_reward + answer_reward
        # else:
        #     return -1.0 + format_reward
    except Exception as e:
        print(f"[DEBUG] Error in compute_all_score: {e}")
        return -1.0, 0.0, 0.0, 0.0
