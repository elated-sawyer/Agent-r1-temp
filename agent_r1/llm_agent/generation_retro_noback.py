"""
Tool generation manager for LLM agents
"""

import torch
import re
import json
import os
import asyncio
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import random
from tqdm.auto import tqdm

try:
    from openai import AsyncOpenAI
    import openai as _openai_module
except ImportError:
    AsyncOpenAI = None
    _openai_module = None

from .tensor_helper import TensorHelper, TensorConfig
# from agent_r1.tool.tool_env import ToolEnv, step, step_batch
from agent_r1.tool.tool_env_retro_noback import ToolEnvRetroNoBack

from verl import DataProto
from verl.utils.tracking import Tracking

@dataclass
class ToolGenerationConfig:
    """Configuration for tool-based generation"""
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_tool_response_length: int  # Renamed from max_obs_length
    num_gpus: int
    # use_parallel_tool_calls: bool = False
    use_batch_tool_calls: bool = False  # New option for batch execution
    tool_call_start: str = "<tool_call>"
    tool_call_end: str = "</tool_call>"
    tool_response_start: str = "<tool_response>"
    tool_response_end: str = "</tool_response>"
    tool_custom_response_template: str = ""
    use_api_model: bool = False
    api_model_name: str = ""
    api_max_concurrency: int = 32
    debug: bool = False
    # "noback" = tool_env_retro_noback.step; "back" = tool_env_retro.step (with back_state)
    tool_env_mode: str = "noback"

class ToolGenerationManager:
    """Manager for handling LLM tool-based generation and interaction"""
    
    def __init__(
        self,
        tokenizer,
        config: ToolGenerationConfig,
        actor_rollout_wg=None,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_tool_response_length=config.max_tool_response_length,  # Renamed
            max_start_length=config.max_start_length,
        ))
        self._api_client = self._build_api_client() if config.use_api_model else None
        _mode = getattr(config, "tool_env_mode", "noback")
        if _mode == "back":
            from agent_r1.tool import tool_env_retro as _env_mod
        elif _mode == "hybrid":
            from agent_r1.tool import tool_env_retro_hybrid as _env_mod
        else:
            from agent_r1.tool import tool_env_retro_noback as _env_mod
        self._step_fn = _env_mod.step
        self._tool_env_mode = _mode

    def _build_api_client(self):
        if AsyncOpenAI is None:
            raise RuntimeError(
                "openai package is required when tool.use_api_model=True but "
                "AsyncOpenAI could not be imported. Install/upgrade `openai` in the env."
            )
        api_key = os.environ.get("pjlab_APImodel_key")
        if not api_key:
            raise RuntimeError(
                "pjlab_APImodel_key is empty. Export it (or set API_KEY_VAR to the "
                "env var that holds your key) before running with tool.use_api_model=True."
            )
        base_url = os.environ.get("pjlab_APImodel_url") or None
        if base_url:
            return AsyncOpenAI(base_url=base_url, api_key=api_key)
        return AsyncOpenAI(api_key=api_key)

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _process_tool_call(self, responses_str) -> Tuple[List[str], List[bool]]:
        """
        Process a list of response strings to extract the first tool call
        while preserving the rest of the string content.
        
        Args:
            responses_str (List[str]): List of response strings potentially containing tool calls
            
        Returns:
            List[str]: Processed responses with only first tool call preserved
        """
        def process_single_response(resp):
            # Look for tool call pattern: <tool_call>tool_name(args)</tool_call>
            tool_pattern = r'<tool_call>(.*?)</tool_call>'
            match = re.search(tool_pattern, resp, re.DOTALL)

            if not match:
                return resp.split(self.tokenizer.eos_token)[0] + self.tokenizer.eos_token, True  # No tool call found
            
            resp = resp.split(self.config.tool_call_end)[0] + self.config.tool_call_end
            # tool_content = match.group(0)
            
            # Replace all subsequent answer tag pairs with their content
            # rest_of_string = resp[match.end():]
            # cleaned_rest = re.sub(r'<tool_call>(.*?)</tool_call>', r'\1', rest_of_string, flags=re.DOTALL)
            
            return resp + self.tokenizer.eos_token, True

            
            # tool_content = match.group(0)
            
            # Replace all subsequent answer tag pairs with their content
            # rest_of_string = resp[match.end():]
            # cleaned_rest = re.sub(r'<tool_call>(.*?)</tool_call>', r'\1', rest_of_string, flags=re.DOTALL)
            
        # Process each response string
        out0 = []
        out1 = []
        for resp in responses_str:
            o0, o1 = process_single_response(resp)
            out0.append(o0)
            out1.append(o1)

        return out0, out1

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to extract tool calls."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        # Extract the first tool call from each response
        responses_str, active_masks = self._process_tool_call(responses_str)
        
        # Tokenize processed responses
        cleaned_token_ids = self._batch_tokenize(responses_str)
        
        return cleaned_token_ids, responses_str, torch.tensor(active_masks, dtype=torch.bool)
    
    def _process_tool_responses(self, tool_responses: List[str]) -> torch.Tensor:
        """Process tool responses to token ids"""
        
        tool_responses_ids = self.tokenizer(
            tool_responses, 
            padding='longest',
            return_tensors='pt'
        )['input_ids']
        
        if tool_responses_ids.shape[1] > self.config.max_tool_response_length:
            print("[WARNING] TOOL RESPONSE TOO LONG, CONSIDER CHANGING YOUR CONFIG")
            tool_responses_ids = tool_responses_ids[:, :self.config.max_tool_response_length]
            
        return tool_responses_ids
    
    def _execute_tool_calls(self, response_strs: List[str], 
                          envs: List[ToolEnvRetroNoBack], 
                          active_mask: torch.Tensor) -> List[str]:
        """Execute tool calls sequentially and return tool responses."""
        # Convert torch tensor to list of booleans if needed
        active_list = active_mask.tolist() if isinstance(active_mask, torch.Tensor) else active_mask
        
        # Initialize result list with empty strings
        tool_responses = [""] * len(response_strs)
        # Process each environment sequentially
        for i, (resp, env, active) in enumerate(zip(response_strs, envs, active_list)):
            if not active:
                continue
                
            # Step the environment using the agent's response

            result = self._step_fn(env, resp)
            
            tool_response = result[0]  # Extract observation from (observation, reward, done, info)
            tool_responses[i] = self.config.tool_custom_response_template.format(tool_response=tool_response)            
        return tool_responses
    
    def _execute_tool_calls_batch(self, response_strs: List[str], 
                                 envs: List[ToolEnvRetroNoBack], 
                                 active_mask: torch.Tensor) -> List[str]:
        """Execute tool calls in batch for tools that support batch operations."""
        # Convert torch tensor to list of booleans
        active_list = active_mask.tolist() if isinstance(active_mask, torch.Tensor) else active_mask
        
        # Filter active environments and responses
        active_envs = []
        active_responses = []
        active_indices = []
        
        for i, (env, resp, active) in enumerate(zip(envs, response_strs, active_list)):
            if active:
                active_envs.append(env)
                active_responses.append(resp)
                active_indices.append(i)
        
        # Initialize result list with empty strings
        tool_responses = [""] * len(response_strs)
        
        if not active_envs:
            return tool_responses
            
        # Use the independent step_batch function for active environments
        batch_results = step_batch(active_envs, active_responses)
        
        # Map results back to original indices
        for idx, result in zip(active_indices, batch_results):
            if result is None:
                tool_responses[idx] = ""
            else:
                tool_response = result[0]  # Extract observation from (observation, reward, done, info)
                tool_responses[idx] = self.config.tool_custom_response_template.format(tool_response=tool_response)
        return tool_responses
    
    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            tool_responses_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            tool_responses_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings
        # return DataProto.from_dict({
        #     'input_ids': new_input_ids[:, -max_len:],
        #     'position_ids': new_position_ids[:, -max_len:],
        #     'attention_mask': new_attention_mask[:, -max_len:]
        # })

    def _info_masked_concatenate_with_padding(self, 
            prompt: torch.Tensor, 
            prompt_with_mask: torch.Tensor, 
            response: torch.Tensor, 
            info: torch.Tensor = None,
            pad_to_left: bool = True
        ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    # def _update_right_side(self, right_side: Dict, 
    #                       cur_responses: torch.Tensor,
    #                       tool_responses_ids: torch.Tensor) -> Dict:
    #     """Update right side state."""
    #     responses = self.tensor_fn.concatenate_with_padding([
    #         right_side['responses'],
    #         cur_responses,
    #         tool_responses_ids
    #     ], pad_to_left=False)
        
    #     batch_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1)
    #     effective_len = batch_len.max()
    #     max_len = min(self.config.max_prompt_length-self.config.max_start_length, effective_len)
    #     # TODO
    #     active_mask = (batch_len <  self.config.max_prompt_length-self.config.max_start_length)
        
    #     return {'responses': responses[:, :max_len]}, active_mask
    
    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          tool_responses_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if tool_responses_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    tool_responses_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        batch_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1)
        effective_len = batch_len.max()
        max_len = min(self.config.max_prompt_length-self.config.max_start_length, effective_len)
        # max_len = min(self.config.max_prompt_length, effective_len)
        active_mask = (batch_len <  self.config.max_prompt_length-self.config.max_start_length)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}, active_mask


    def _generate_with_api(self, active_batch: DataProto) -> DataProto:
        """Generate responses using an external API model instead of local vLLM."""
        input_ids = active_batch.batch['input_ids']
        batch_size = input_ids.shape[0]

        # Decode input_ids back to text prompts, stripping left-padding first
        prompts = []
        for ids in input_ids:
            non_pad_ids = ids[ids != self.tokenizer.pad_token_id]
            prompts.append(self.tokenizer.decode(non_pad_ids, skip_special_tokens=False))

        if self.config.debug:
            prompt_lens = [len(p) for p in prompts]
            print(f"[DEBUG API] Sending {batch_size} prompts to {self.config.api_model_name} | "
                  f"len min={min(prompt_lens)} max={max(prompt_lens)} avg={sum(prompt_lens)/len(prompt_lens):.0f}")
            for i, p in enumerate(prompts[:2]):
                print(f"[DEBUG API] Prompt[{i}] (len={len(p)}): {p[:]}...") # 500

        sem = asyncio.Semaphore(self.config.api_max_concurrency)

        _RETRYABLE = tuple(filter(None, [
            getattr(_openai_module, 'RateLimitError', None),
            getattr(_openai_module, 'APITimeoutError', None),
            getattr(_openai_module, 'APIConnectionError', None),
            getattr(_openai_module, 'InternalServerError', None),
        ])) if _openai_module else ()
        _MAX_RETRIES = 10
        _BASE_WAIT = 5.0
        _MAX_WAIT = 120.0

        async def _call_api(prompt: str) -> str:
            """Call external API with concurrency limit and exponential backoff."""
            async with sem:
                last_exc = None
                for attempt in range(_MAX_RETRIES):
                    try:
                        response = await self._api_client.chat.completions.create(
                            model=self.config.api_model_name,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=self.config.max_response_length,
                            temperature=0.0,
                            top_p=1.0,
                            n=1,
                        )
                        return response.choices[0].message.content or ""
                    except _RETRYABLE as exc:
                        last_exc = exc
                        wait = min(_BASE_WAIT * (2 ** attempt), _MAX_WAIT)
                        wait += random.uniform(0, wait * 0.25)
                        print(f"[RETRY {attempt+1}/{_MAX_RETRIES}] {type(exc).__name__}: {exc} | "
                              f"waiting {wait:.1f}s before retry")
                        await asyncio.sleep(wait)
                    except Exception as exc:
                        raise RuntimeError(f"API request failed (non-retryable): {type(exc).__name__}: {exc}") from None
                raise RuntimeError(
                    f"API request failed after {_MAX_RETRIES} retries: {type(last_exc).__name__}: {last_exc}"
                )

        async def _call_all():
            return await asyncio.gather(*[_call_api(p) for p in prompts], return_exceptions=True)

        # Run async calls
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        try:
            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    api_results = pool.submit(asyncio.run, _call_all()).result()
            else:
                api_results = asyncio.run(_call_all())
        except Exception as exc:
            raise RuntimeError(
                f"API batch generation failed before decoding: {type(exc).__name__}: {exc}"
            ) from None

        response_texts = []
        for result in api_results:
            if isinstance(result, Exception):
                print(f"[WARNING] {result}")
                response_texts.append("")
            else:
                response_texts.append(result)

        if self.config.debug:
            resp_lens = [len(r) for r in response_texts]
            print(f"[DEBUG API] Received {len(response_texts)} responses | "
                  f"len min={min(resp_lens)} max={max(resp_lens)} avg={sum(resp_lens)/len(resp_lens):.0f}")
            for i, r in enumerate(response_texts[:2]):
                print(f"[DEBUG API] Response[{i}] (len={len(r)}): {r[:]}...") # 500

        # Tokenize responses to match the tensor format from _generate_with_gpu_padding
        response_ids = self.tokenizer(
            response_texts,
            add_special_tokens=False,
            return_tensors='pt',
            padding='longest',
        )['input_ids'].to(torch.long)

        # Pad/truncate to max_response_length to match expected shape.
        # Note: when all response_texts are empty strings, the tokenizer returns a
        # tensor of shape [batch_size, 0] whose dtype may default to float, which
        # later breaks tokenizer.batch_decode (expects integer ids). We therefore
        # explicitly force dtype=torch.long here and handle the empty-seq case.
        max_resp_len = self.config.max_response_length
        if response_ids.shape[1] == 0:
            response_ids = torch.full(
                (batch_size, max_resp_len),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
            )
        elif response_ids.shape[1] < max_resp_len:
            pad = torch.full(
                (batch_size, max_resp_len - response_ids.shape[1]),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
            )
            response_ids = torch.cat([response_ids, pad], dim=1)
        elif response_ids.shape[1] > max_resp_len:
            response_ids = response_ids[:, :max_resp_len]

        result = DataProto.from_dict({'responses': response_ids})
        result.meta_info = active_batch.meta_info if hasattr(active_batch, 'meta_info') and active_batch.meta_info else {}
        return result

    def run_llm_loop(self, gen_batch, envs: List[Any] = None,
                    initial_input_ids: torch.Tensor = None) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        # original_right_side = {'responses': initial_input_ids[:, []]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        batch_size = gen_batch.batch['input_ids'].shape[0]
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        pad_size = batch_size - len(envs)
        if pad_size > 0:
            active_mask[-pad_size:] = 0
        turns = torch.zeros(batch_size, dtype=torch.int32)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Main generation loop
        desc = "Validation LLM turns" if self.is_validation else "Train LLM turns"
        pbar = tqdm(range(self.config.max_turns), desc=desc, leave=True)
        for step in pbar:
            pbar.set_postfix(active=f"{active_mask.sum().item()}/{batch_size}")
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            gen_output = self._generate_with_api(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str, new_active_masks = self._postprocess_responses(gen_output.batch['responses'])
            # print('responses_ids: ', responses_ids.shape)
            # print('active_mask: ', active_mask.sum())

            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            active_mask[active_mask.clone()] = new_active_masks

            turns[active_mask] += 1

            if self.config.use_batch_tool_calls:
                # Use batch execution for tool calls
                tool_responses = self._execute_tool_calls_batch(responses_str, envs, active_mask)
            else:
                # Use sequential execution for tool calls
                tool_responses = self._execute_tool_calls(responses_str, envs, active_mask)

            for ii in range(len(envs)):
                if not active_mask[ii]:
                    continue
                env = envs[ii]

                # In noback mode, back_flag=True is a terminal signal (there is
                # no back_state tool, so the trajectory is dead). In back/hybrid
                # modes, back_flag=True means "the model MUST call back_state
                # next"; we must keep the trajectory active or the rescue path
                # never fires.
                if env.back_flag and self._tool_env_mode == "noback":
                    active_mask[ii] = False

                for key, value in env.unsolved_dict.items():
                    if not value:
                        active_mask[ii] = False
                        break

            active_num_list.append(active_mask.sum().item())
            # for ii in range(len(active_mask)):
            #     if not active_mask[ii]:
            #         tool_responses[ii] = ""
            tool_responses_ids = self._process_tool_responses(tool_responses)
            # for ii in range(len(active_mask)):
            #     if not active_mask[ii]:
            #         tool_responses_ids[ii, :] = self.tokenizer.pad_token_id
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                tool_responses_ids
            )
            original_right_side, update_active_masks = self._update_right_side(
                original_right_side,
                responses_ids,
                tool_responses_ids
            )

            for ii in range(len(update_active_masks)):
                if not update_active_masks[ii]:
                    active_mask[ii] = False


        
        pbar.set_postfix(active=f"{active_mask.sum().item()}/{batch_size}")
        pbar.close()

        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        original_right_side['turns'] = turns
        
        # Save trajectory and return final output
        return self._compose_final_output(original_left_side, original_right_side, meta_info)


    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']

        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)

        return final_output
