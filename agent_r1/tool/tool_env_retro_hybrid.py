"""
Hybrid tool environment for retrosynthesis.

Design summary:
  * Starts in "Phase A" (identical to ToolEnvRetroNoBack): model can only call
    `single_step_retro` and `select_reaction`, and is pushed to go deep.
  * When `current_step >= maxstep` (or a MolFail dead-end), env sets
    `back_flag = True` -> "Phase B": model must use `back_state` to rescue.
  * On successful `back_state`, `back_flag` is reset to False and execution
    returns to Phase A. The A <-> B cycle can repeat until the trajectory
    finishes naturally (all unsolved molecules solved, or max_turns exhausted).

Differences vs ToolEnvRetro (every-step backtracking):
  * `back_state` is gated by `back_flag` (Phase B only), so the model cannot
    shallow-chase by voluntarily retreating every turn.
  * `force_noloop` filtering from ToolEnvRetroNoBack is preserved.

Differences vs ToolEnvRetroNoBack (no backtracking at all):
  * `back_state` is registered and handled as a real tool, so hitting maxstep
    is recoverable rather than terminal.
"""

import re
import json
import random
import traceback
from typing import Dict, List, Any, Tuple, Optional
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from rdkit import Chem
import itertools
from agent_r1.tool.tool_base import Tool


def step(env: 'ToolEnvRetroHybrid', action_text: str):
    """Execute one step of environment interaction."""
    env.steps_taken += 1
    action = env.extract_tool_call(action_text)

    if action == env.INVALID_ACTION:
        result = "Invalid tool call format. Please use the format: \n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {params_json}}\n</tool_call> ."
        reward = env.PENALTY_FOR_INVALID
        env._update_tracking_variables(
            response=action_text,
            action=action,
            action_is_valid=False,
            action_is_effective=False,
            reward=reward
        )
        return result, reward, False, {"action_is_valid": False, "action_is_effective": False}

    tool_name = action["tool"]
    tool_args = action["args"]

    if tool_name not in env.tool_map:
        result = f"Unknown tool: {tool_name}"
        reward = env.PENALTY_FOR_INEFFECTIVE
        env._update_tracking_variables(
            response=action_text,
            action=action,
            action_is_valid=True,
            action_is_effective=False,
            reward=reward
        )
        return result, reward, False, {"action_is_valid": True, "action_is_effective": False}

    is_valid, error_msg = env.check_tool_applicability(tool_name)
    if not is_valid:
        result = f"Invalid choice of tool '{tool_name}': {error_msg}"
        reward = env.PENALTY_FOR_INEFFECTIVE
        env._update_tracking_variables(
            response=action_text,
            action=action,
            action_is_valid=True,
            action_is_effective=False,
            reward=reward
        )
        return result, reward, False, {"action_is_valid": True, "action_is_effective": False}

    tool = env.tool_map[tool_name]

    is_valid, error_msg = tool.validate_args(tool_args)
    if not is_valid:
        result = f"Invalid arguments for tool '{tool_name}': {error_msg}"
        reward = env.PENALTY_FOR_INEFFECTIVE
        env._update_tracking_variables(
            response=action_text,
            action=action,
            action_is_valid=True,
            action_is_effective=False,
            reward=reward
        )
        return result, reward, False, {"action_is_valid": True, "action_is_effective": False}

    env_tool_args, is_valid, error_msg = env.wrap_tool_args(tool_name, tool_args)
    if not is_valid:
        result = f"Invalid arguments for tool '{tool_name}': {error_msg}"
        reward = env.PENALTY_FOR_INEFFECTIVE
        env._update_tracking_variables(
            response=action_text,
            action=action,
            action_is_valid=True,
            action_is_effective=False,
            reward=reward
        )
        return result, reward, False, {"action_is_valid": True, "action_is_effective": False}

    try:
        result = tool.execute(env_tool_args)
        reward = tool.calculate_reward(env_tool_args, result)
        if 'error' in result:
            env_result = f"Error executing tool '{tool_name}': {result['error']}"
            env._update_tracking_variables(
                response=action_text,
                action=action,
                action_is_valid=True,
                action_is_effective=False,
                reward=reward
            )
            return env_result, reward, False, {"action_is_valid": True, "action_is_effective": False}
        env_result = env._update_state_variables_message(
            tool_name=tool_name,
            tool_args=tool_args,
            env_tool_args=env_tool_args,
            result=result
        )

        env.tool_history.append({
            "tool": tool_name,
            "args": tool_args,
            "result": env_result
        })

        done = env.steps_taken >= env.max_turns

        env._update_tracking_variables(
            response=action_text,
            action=action,
            action_is_valid=True,
            action_is_effective=True,
            reward=reward
        )

        return env_result, reward, done, {"action_is_valid": True, "action_is_effective": True}
    except Exception as e:
        error_trace = traceback.format_exc()
        result = f"Error executing tool '{tool_name}': {str(e)}"
        reward = env.PENALTY_FOR_INEFFECTIVE

        env._update_tracking_variables(
            response=action_text,
            action=action,
            action_is_valid=True,
            action_is_effective=False,
            reward=reward
        )

        return result, reward, False, {"action_is_valid": True, "action_is_effective": False}


def canonicalize_smiles_clear_map(smiles, return_max_frag=True):
    mol = Chem.MolFromSmiles(smiles, sanitize=not False)
    if mol is not None:
        [
            atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms()
            if atom.HasProp('molAtomMapNumber')
        ]
        try:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        except:
            return ''

        return smi
    else:
        return ''


def process_chira(mol):
    if '@' not in mol:
        return [mol]
    mol_list = [mol]
    for m in mol_list:
        new_m = canonicalize_smiles_clear_map(m)
        if new_m not in mol_list:
            mol_list.append(new_m)
    return mol_list


Instruction = "You are a professional organic chemist, skilled at finding pathways to synthesize novel molecules. Given a target molecule, your task is to find a pathway where the target can be eventually synthesized using ingredient molecules, and each step of the pathway is a viable chemical reaction. You can conduct 4 actions: \n1 'Call'. Select an unsolved molecule and I'll provide you with several reactions to synthesize it. Note that the reactions may be incorrect and the reactants is marked as 'available' or 'unavailable'. The unavailable molecules have to be solved further. \n2 'Select'. After the action Call one-step model, you can use this action to select a viable reaction. If the reaction has unavailable reactants, please make sure they are easier to synthesize. \n3 'Back'. If you realize that you can take better actions at a history state, you can go back to it. \n4 'Exit'. If you have found at least one pathway to synthesize the target molecule and you can't find a better one, you can use this action to finish the task. \nThe maximum number of reaction steps you can use is {n}.\n"
Success_select = "Successfully select reaction {i}-{j}-{k} to synthesize molecule {mol} . The state goes to: \n"
Statei = "Molecule state {i}: Unsolved molecules: "
Statei_step = "\nTo reach this state, at least {n} reaction steps are used."
Molm = "\nMolecule {i}-{m}: {mol}"
MolFail = "\nPathway fails because molecule {i}-{j}: {mol} can not be synthesized."
StepFail = "\nPathway fails because the maximum number of reaction steps {n} has been reached but some molecules are still unsolved."
Statei_end = "Molecule state {i}: No unsolved molecules."

ToSelect = "\nPlease determine which of these reactions are possible. Then use the tool 'select_reaction' to choose the best reaction whose reactants are available or not available but easy to synthesize."
ToCall = "\nPlease select the molecule with the highest synthesis difficulty from the unsolved molecules. Use the tool 'single_step_retro' to synthesize it."
FailBack = "\nPlease use tool 'back_state' to go to a history state and continue to search."
SuccessBack = "\nYou have successfully found a pathway to synthesize the target molecule {mol}. Please use tool 'back_state' to go to a history molecule or reaction state and search for a shorter pathway."
GoBackMolecule = "Go back to the molecule state {i}: Unsolved molecules: "
GoBackReaction = "Go back to the reaction state {i}-{j}: Possible reactions to synthesize molecule {i}-{j}: {mol} are as follows: "

Statei_j = "Reaction state {i}-{j}: Possible reactions to synthesize molecule {i}-{j}: {mol} are as follows: "
Reactionk = "\nReaction {i}-{j}-{k}: "
ReactionNone = "\nNo reactions."
Singlereactant = "{mol} ({avail})"

Action_Call = "Call. Synthesize Mol {i}-{m}"
Action_Select = "Select. Select Reaction {i}-{j}-{k}"
Action_Back2i = "Back. Back to State {i}\n"
Action_Back2i_j = "Back. Back to State {i}-{j}\n"
Action_Exit = "Exit."


class ToolEnvRetroHybrid:
    """
    Hybrid retrosynthesis env: noback-style until a dead-end, then one rescue
    via `back_state`, then back to noback-style.

    `back_flag` is the gate:
      * False (Phase A): only `single_step_retro`/`select_reaction` usable
      * True  (Phase B): only `back_state` usable
    It is flipped True on:
      * maxstep reached after a successful select_reaction, or
      * MolFail (a chosen molecule returns no viable reactions), or
      * full synthesis of the target (legacy; trajectory ends naturally
        via unsolved_dict emptying before the model can act)
    and flipped False on a successful `back_state`.
    """
    INVALID_ACTION = {"tool": "invalid", "args": {}}
    PENALTY_FOR_INVALID = -0.1
    PENALTY_FOR_INEFFECTIVE = -0.05

    def __init__(self, tools: List[Tool] = None, max_turns: int = 10, maxstep: int = 50, topk: int = 5, shuffle: bool = False, force_noloop: bool = False):
        self.tools = tools or []
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.tool_desc = [tool.get_description() for tool in self.tools]
        self.max_turns = max_turns
        self.maxstep = maxstep
        self.topk = topk
        self.max_reaction = 30
        self.shuffle = shuffle
        self.force_noloop = force_noloop
        self.target = None
        self.reset_tracking_variables()

    def tools_format_func(self) -> str:
        template = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""
        tools = "\n".join([f"{json.dumps(tool.get_description(), ensure_ascii=False)}" for tool in self.tools])
        return template.format(tools=tools)

    def reset_tracking_variables(self):
        self.rewards = []
        self.tool_history = []
        self.steps_taken = 0
        self._actions = []
        self._actions_valid = []
        self._actions_effective = []

        if self.target:
            self.idx = 1
            self.state_relation = [[], [], []]
            self.unsolved_dict = {'0': [self.target]}
            self.reaction_dict = {}
            self.mol_dict = {}
            self.step_dict = {'0': 0}
            self.current_state = 0
            self.current_molid = -1
            self.count_onestep = 0
            self.count_conversion = 0
            self.count_totaltry = 0
            self.back_flag = False
        else:
            self.idx = 0
            self.state_relation = [[], [], []]
            self.unsolved_dict = {}
            self.reaction_dict = {}
            self.mol_dict = {}
            self.step_dict = {}
            self.current_state = 0
            self.current_molid = -1
            self.count_onestep = 0
            self.count_conversion = 0
            self.count_totaltry = 0
            self.back_flag = False

    def get_tracking_variables(self) -> Dict:
        return {
            "rewards": self.rewards,
            "total_reward": sum(self.rewards),
            "steps_taken": self.steps_taken,
            "tool_history": self.tool_history,
            "actions": self._actions,
            "actions_valid": self._actions_valid,
            "actions_effective": self._actions_effective,
            "idx": self.idx,
            "state_relation": self.state_relation,
            "unsolved_dict": self.unsolved_dict,
            "reaction_dict": self.reaction_dict,
            "mol_dict": self.mol_dict,
            "step_dict": self.step_dict,
            "current_state": self.current_state,
            "current_molid": self.current_molid,
            "count_onestep": self.count_onestep,
            "count_conversion": self.count_conversion,
            "count_totaltry": self.count_totaltry,
            "back_flag": self.back_flag,
        }

    def _update_tracking_variables(
            self,
            response: str,
            action: Any,
            action_is_valid: bool,
            action_is_effective: bool,
            reward: float,
        ):
        self._actions.append(response)
        if action_is_valid:
            self._actions_valid.append(action)
        else:
            self._actions_valid.append(None)
        if action_is_effective:
            self._actions_effective.append(action)
        else:
            self._actions_effective.append(None)

        self.rewards.append(reward)

    def extract_tool_call(self, text: str) -> Dict:
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'

        tool_call_match = re.search(tool_call_pattern, text, re.DOTALL)

        if not tool_call_match:
            return self.INVALID_ACTION

        try:
            tool_call_json = tool_call_match.group(1).strip()
            tool_call_data = json.loads(tool_call_json)

            if "name" not in tool_call_data:
                return self.INVALID_ACTION

            tool_name = tool_call_data["name"]
            tool_args = tool_call_data.get("arguments", {})
            if "reaction" in tool_args:
                reaction_split = tool_args["reaction"].split('-')
                reaction_split_u = [str(int(r)) for r in reaction_split]
                reaction_u = '-'.join(reaction_split_u)
                if tool_args["reaction"] != reaction_u:
                    print("Different ID: ", tool_args["reaction"])
                tool_args["reaction"] = reaction_u

            if "molecule" in tool_args:
                molecule_split = tool_args["molecule"].split('-')
                molecule_split_u = [str(int(r)) for r in molecule_split]
                molecule_u = '-'.join(molecule_split_u)
                if tool_args["molecule"] != molecule_u:
                    print("Different ID: ", tool_args["molecule"])
                tool_args["molecule"] = molecule_u

            return {"tool": tool_name, "args": tool_args}
        except json.JSONDecodeError:
            return self.INVALID_ACTION
        except Exception:
            return self.INVALID_ACTION

    def check_tool_applicability(self, tool_name):
        """Gate each tool by (current_molid, back_flag).

        Phase A (back_flag=False):
          * tools[0] single_step_retro: requires current_molid == -1 (fresh state)
          * tools[1] select_reaction:   requires current_molid != -1 (just called)
          * tools[2] back_state:        REJECTED (only usable in Phase B)

        Phase B (back_flag=True):
          * tools[0], tools[1]:         REJECTED
          * tools[2] back_state:        allowed
        """
        if tool_name == self.tools[0].name:
            if self.current_molid != -1 or self.back_flag:
                error_msg = f"{tool_name} cannot be used at current state."
                return False, error_msg
            else:
                return True, ""
        elif tool_name == self.tools[1].name:
            if self.current_molid == -1 or self.back_flag:
                error_msg = f"{tool_name} cannot be used at current state."
                return False, error_msg
            else:
                return True, ""
        elif tool_name == self.tools[2].name:
            if not self.back_flag:
                error_msg = (
                    f"{tool_name} can only be used after reaching the maximum number "
                    f"of reaction steps or a dead-end. Please continue with "
                    f"'single_step_retro' or 'select_reaction'."
                )
                return False, error_msg
            else:
                return True, ""
        else:
            raise ValueError(f"Unknown tool {tool_name}.")

    def _process_action_call(self, value):
        value_list = value.split('-')
        unsolved_list = self.unsolved_dict[str(self.current_state)]
        if len(value_list) != 2 or not value_list[0].isdigit() or not value_list[1].isdigit():
            error_msg = f"Parameter molecule has invalid value, should be {self.current_state}-0 ~ {self.current_state}-{len(unsolved_list)-1} ."
            is_valid = False
            return error_msg, is_valid
        if int(value_list[0]) != self.current_state:
            error_msg = f"Parameter molecule has invalid value, should be {self.current_state}-0 ~ {self.current_state}-{len(unsolved_list)-1} ."
            is_valid = False
            return error_msg, is_valid
        if int(value_list[1]) >= len(unsolved_list):
            error_msg = f"Parameter molecule has invalid value, should be {self.current_state}-0 ~ {self.current_state}-{len(unsolved_list)-1} ."
            is_valid = False
            return error_msg, is_valid

        return "", True

    def _process_action_select(self, value):
        value_list = value.split('-')
        reactionid = str(self.current_state) + '-' + str(self.current_molid)
        if len(value_list) != 3 or not value_list[0].isdigit() or not value_list[1].isdigit() or not value_list[2].isdigit():
            error_msg = f"Parameter reaction has invalid value, should be {self.current_state}-{self.current_molid}-0 ~ {self.current_state}-{self.current_molid}-{len(self.reaction_dict[reactionid])-1} ."
            is_valid = False
            return error_msg, is_valid
        if int(value_list[0]) != self.current_state:
            error_msg = f"Parameter reaction has invalid value, should be {self.current_state}-{self.current_molid}-0 ~ {self.current_state}-{self.current_molid}-{len(self.reaction_dict[reactionid])-1} ."
            is_valid = False
            return error_msg, is_valid
        if int(value_list[1]) != self.current_molid:
            error_msg = f"Parameter reaction has invalid value, should be {self.current_state}-{self.current_molid}-0 ~ {self.current_state}-{self.current_molid}-{len(self.reaction_dict[reactionid])-1} ."
            is_valid = False
            return error_msg, is_valid
        if reactionid not in self.reaction_dict:
            error_msg = f"Parameter reaction has invalid value, should be {self.current_state}-{self.current_molid}-0 ~ {self.current_state}-{self.current_molid}-{len(self.reaction_dict[reactionid])-1} ."
            is_valid = False
            return error_msg, is_valid
        if int(value_list[2]) >= len(self.reaction_dict[reactionid]):
            error_msg = f"Parameter reaction has invalid value, should be {self.current_state}-{self.current_molid}-0 ~ {self.current_state}-{self.current_molid}-{len(self.reaction_dict[reactionid])-1} ."
            is_valid = False
            return error_msg, is_valid

        return "", True

    def _process_action_back(self, value):
        value_list = value.split('-')
        if len(value_list) == 1:
            if not value_list[0].isdigit():
                error_msg = f"Parameter state has invalid value {value} ."
                is_valid = False
                return error_msg, is_valid
            if value not in self.unsolved_dict:
                error_msg = f"Parameter state has invalid value {value} because the selected state is not in the history."
                is_valid = False
                return error_msg, is_valid
            return "", True

        elif len(value_list) == 2:
            if not value_list[0].isdigit() and not value_list[1].isdigit():
                error_msg = f"Parameter state has invalid value {value} ."
                is_valid = False
                return error_msg, is_valid
            if value not in self.reaction_dict:
                error_msg = f"Parameter state has invalid value {value} because the selected state is not in the history."
                is_valid = False
                return error_msg, is_valid

            return "", True

        else:
            error_msg = f"Parameter state has invalid value {value} ."
            is_valid = False
            return error_msg, is_valid

    def wrap_tool_args(self, tool_name: str, tool_args: Dict) -> Tuple[Dict, bool, str]:
        if tool_name == self.tools[0].name:
            error_msg, is_valid = self._process_action_call(tool_args["molecule"])
            if not is_valid:
                return {}, is_valid, error_msg
            else:
                value_list = tool_args["molecule"].split('-')
                current_mol = self.unsolved_dict[str(self.current_state)][int(value_list[1])]
                if tool_args["molecule"] in self.reaction_dict:
                    env_tool_args = {
                        "molecule": current_mol,
                        "exist": True,
                        "reaction_list": self.reaction_dict[tool_args["molecule"]],
                        "topk": self.topk,
                        "shuffle": self.shuffle
                    }
                elif current_mol in self.mol_dict:
                    env_tool_args = {
                        "molecule": current_mol,
                        "exist": True,
                        "reaction_list": self.mol_dict[current_mol],
                        "topk": self.topk,
                        "shuffle": self.shuffle
                    }
                else:
                    env_tool_args = {
                        "molecule": current_mol,
                        "exist": False,
                        "reaction_list": [],
                        "topk": self.topk,
                        "shuffle": self.shuffle
                    }
                return env_tool_args, is_valid, ""
        elif tool_name == self.tools[1].name:
            error_msg, is_valid = self._process_action_select(tool_args["reaction"])
            if not is_valid:
                return {}, is_valid, error_msg
            else:
                return tool_args, is_valid, ""
        elif tool_name == self.tools[2].name:
            error_msg, is_valid = self._process_action_back(tool_args["state"])
            if not is_valid:
                return {}, is_valid, error_msg
            else:
                return tool_args, is_valid, ""

        else:
            raise ValueError(f"Unknown tool {tool_name}.")

    def _update_state_variables_message(self, tool_name, tool_args, env_tool_args, result):
        if tool_name == self.tools[0].name:
            try:
                reaction_list = result['results']
                self.current_molid = int(tool_args['molecule'].split('-')[1])

                if self.force_noloop:
                    current_mol_t = self.unsolved_dict[str(self.current_state)][self.current_molid]
                    current_unsolved_t = deepcopy(self.unsolved_dict[str(self.current_state)])
                    current_unsolved_t.remove(current_mol_t)
                    current_unsolved_prod_t = [[m] for m in current_unsolved_t]
                    reaction_list_left = []
                    for reaction_item in reaction_list:
                        current_unsolved_prod_ti = deepcopy(current_unsolved_prod_t)
                        current_unsolved_ti = deepcopy(current_unsolved_t)
                        for reactant_t in reaction_item:
                            if reactant_t[1] == "unavailable":
                                reactant_t_list = process_chira(reactant_t[0])
                                for r in reactant_t_list:
                                    if r in current_unsolved_ti:
                                        break
                                else:
                                    current_unsolved_ti += reactant_t_list
                                    current_unsolved_prod_ti.append(reactant_t_list)

                        if current_unsolved_ti:
                            possible_combine_ti = list(itertools.product(*current_unsolved_prod_ti))

                            next_state_id_ti = -1
                            for p in possible_combine_ti:
                                for key_u, value_u in self.unsolved_dict.items():
                                    if set(p) == set(value_u):
                                        next_state_id_ti = int(key_u)
                                    if next_state_id_ti >= 0:
                                        break
                                if next_state_id_ti >= 0:
                                    break
                            else:
                                reaction_list_left.append(reaction_item)
                        else:
                            reaction_list_left.append(reaction_item)

                else:
                    reaction_list_left = reaction_list

                if reaction_list_left:
                    if tool_args['molecule'] not in self.reaction_dict:
                        self.reaction_dict[tool_args['molecule']] = deepcopy(reaction_list_left)
                    if env_tool_args['molecule'] not in self.mol_dict:
                        self.count_onestep += 1
                        current_mol_list = process_chira(env_tool_args['molecule'])
                        for m in current_mol_list:
                            self.mol_dict[m] = deepcopy(reaction_list)
                    current_message = Statei_j.format(i=self.current_state, j=self.current_molid, mol=env_tool_args['molecule'])
                    for idy, reaction in enumerate(reaction_list_left[:self.max_reaction]):
                        current_message += Reactionk.format(i=self.current_state, j=self.current_molid, k=idy)
                        for idz, reactant in enumerate(reaction):
                            if idz > 0:
                                current_message += ' + '
                            current_message += Singlereactant.format(mol=reactant[0], avail=reactant[1])
                    current_message += ToSelect

                else:
                    if tool_args['molecule'] not in self.reaction_dict:
                        self.reaction_dict[tool_args['molecule']] = []
                    if env_tool_args['molecule'] not in self.mol_dict:
                        self.count_onestep += 1
                        current_mol_list = process_chira(env_tool_args['molecule'])
                        for m in current_mol_list:
                            self.mol_dict[m] = []
                    current_message = Statei_j.format(i=self.current_state, j=self.current_molid, mol=env_tool_args['molecule'])
                    current_message += ReactionNone
                    current_message += MolFail.format(i=self.current_state, j=self.current_molid, mol=env_tool_args['molecule'])
                    # Dead-end: no viable reactions for the chosen molecule -> enter Phase B
                    current_message += FailBack
                    self.back_flag = True

                return current_message

            except Exception as e:
                raise ValueError(str(e))
        elif tool_name == self.tools[1].name:
            try:
                value = tool_args["reaction"]
                value_list = value.split('-')
                reactionid = str(self.current_state) + '-' + str(self.current_molid)
                current_reaction = self.reaction_dict[reactionid][int(value_list[2])]

                current_mol = self.unsolved_dict[str(self.current_state)][self.current_molid]
                current_unsolved = deepcopy(self.unsolved_dict[str(self.current_state)])
                current_unsolved.remove(current_mol)
                current_unsolved_prod = [[m] for m in current_unsolved]
                for reactant in current_reaction:
                    if reactant[1] == "unavailable":
                        reactant_list = process_chira(reactant[0])
                        for r in reactant_list:
                            if r in current_unsolved:
                                break
                        else:
                            current_unsolved += reactant_list
                            current_unsolved_prod.append(reactant_list)

                if current_unsolved:
                    possible_combine = list(itertools.product(*current_unsolved_prod))

                    next_state_id = -1
                    for p in possible_combine:
                        for key_u, value_u in self.unsolved_dict.items():
                            if set(p) == set(value_u):
                                next_state_id = int(key_u)
                            if next_state_id >= 0:
                                break
                        if next_state_id >= 0:
                            break

                    if next_state_id == -1:
                        next_state_id = self.idx
                        self.idx += 1
                        current_step = self.step_dict[str(self.current_state)] + 1
                        self.step_dict[str(next_state_id)] = current_step
                        self.state_relation[0].append(self.current_state)
                        self.state_relation[1].append(next_state_id)
                        self.state_relation[2].append([value])
                        unsolved_list = list(possible_combine[0])
                        self.unsolved_dict[str(next_state_id)] = deepcopy(unsolved_list)
                        self.current_state = next_state_id

                    else:
                        current_step = self.step_dict[str(self.current_state)] + 1
                        next_step = self.step_dict[str(next_state_id)]
                        if current_step < next_step:
                            self.step_dict[str(next_state_id)] = current_step
                            current_update = [next_state_id]
                            next_update = []
                            step_update = current_step + 1
                            while current_update:
                                for idstate in current_update:
                                    for ii in range(len(self.state_relation[0])):
                                        if self.state_relation[0][ii] == idstate:
                                            idstate_next = self.state_relation[1][ii]
                                            if step_update < self.step_dict[str(idstate_next)]:
                                                self.step_dict[str(idstate_next)] = step_update
                                                next_update.append(step_update)
                                current_update = next_update
                                next_update = []
                                step_update += 1
                        else:
                            current_step = next_step

                        for ii in range(len(self.state_relation[0])):
                            if self.state_relation[0][ii] == self.current_state and self.state_relation[1][ii] == next_state_id:
                                if value not in self.state_relation[2][ii]:
                                    self.state_relation[2][ii].append(value)
                                break
                        else:
                            self.state_relation[0].append(self.current_state)
                            self.state_relation[1].append(next_state_id)
                            self.state_relation[2].append([value])

                        unsolved_list = self.unsolved_dict[str(next_state_id)]
                        self.current_state = next_state_id

                    current_message = Success_select.format(i=value_list[0], j=value_list[1], k=value_list[2], mol=current_mol)
                    current_message += Statei.format(i=self.current_state)
                    for idy, mol in enumerate(unsolved_list):
                        current_message += Molm.format(i=self.current_state, m=idy, mol=mol)
                    current_message += Statei_step.format(n=current_step)
                    if current_step >= self.maxstep:
                        # Hit maxstep -> enter Phase B (rescue via back_state).
                        current_message += StepFail.format(n=self.maxstep)
                        current_message += FailBack
                        self.back_flag = True
                    else:
                        current_message += ToCall


                else:
                    current_step = self.step_dict[str(self.current_state)] + 1
                    for key, value in self.unsolved_dict.items():
                        if not value:
                            next_state_id = int(key)
                            if current_step < self.step_dict[str(next_state_id)]:
                                self.step_dict[str(next_state_id)] = current_step
                            else:
                                current_step = self.step_dict[str(next_state_id)]
                            for ii in range(len(self.state_relation[0])):
                                if self.state_relation[0][ii] == self.current_state and self.state_relation[1][ii] == next_state_id:
                                    if value not in self.state_relation[2][ii]:
                                        self.state_relation[2][ii].append(value)
                                    break
                            else:
                                self.state_relation[0].append(self.current_state)
                                self.state_relation[1].append(next_state_id)
                                self.state_relation[2].append([value])
                            unsolved_list = []
                            self.current_state = next_state_id
                            break
                    else:
                        next_state_id = self.idx
                        self.idx += 1
                        self.step_dict[str(next_state_id)] = current_step

                        self.state_relation[0].append(self.current_state)
                        self.state_relation[1].append(next_state_id)
                        self.state_relation[2].append([value])
                        unsolved_list = []
                        self.unsolved_dict[str(next_state_id)] = []
                        self.current_state = next_state_id
                    current_message = Success_select.format(i=value_list[0], j=value_list[1], k=value_list[2], mol=current_mol)
                    current_message += Statei_end.format(i=self.current_state, n=current_step)
                    current_message += Statei_step.format(n=current_step)
                    # Target fully synthesized. We leave back_flag=False and DO NOT
                    # append SuccessBack: the trajectory terminates naturally via
                    # unsolved_dict having an empty entry (the generation loop
                    # flips active_mask off). This matches the hybrid design
                    # requirement: find one path, stop - don't chase shorter.
                self.current_molid = -1

                return current_message

            except Exception as e:
                raise ValueError(str(e))
        elif tool_name == self.tools[2].name:
            # back_state: jump the env pointer to a history state, reset Phase A.
            # Validity checks (state exists in history, not already at maxstep,
            # etc.) raise ValueError which the outer step() converts into a
            # reward-penalised ineffective tool call. back_flag stays True in
            # that case so the model can retry with a different target state.
            try:
                value = tool_args["state"]
                value_list = value.split('-')
                if len(value_list) == 1:
                    target_state = int(value_list[0])
                    current_unsolved = self.unsolved_dict[value]
                    current_step = self.step_dict[value]
                    if not current_unsolved:
                        raise ValueError("You can't go back to a molecule state with no unsolved molecules.")
                    elif current_step >= self.maxstep:
                        raise ValueError(f"You can't go back to a molecule state where the maximum number of reaction steps {self.maxstep} has been reached.")
                    else:
                        self.current_state = target_state
                        self.current_molid = -1
                        current_message = GoBackMolecule.format(i=self.current_state)
                        for idy, mol in enumerate(current_unsolved):
                            current_message += Molm.format(i=self.current_state, m=idy, mol=mol)
                        current_message += Statei_step.format(n=current_step)
                        current_message += ToCall
                elif len(value_list) == 2:
                    target_state = int(value_list[0])
                    target_molid = int(value_list[1])
                    reaction_list = self.reaction_dict[value]
                    if not reaction_list:
                        raise ValueError("You can't go back to a reaction state where the molecule cannot be synthesized.")
                    else:
                        self.current_state = target_state
                        self.current_molid = target_molid
                        current_mol = self.unsolved_dict[str(self.current_state)][self.current_molid]
                        current_message = GoBackReaction.format(i=self.current_state, j=self.current_molid, mol=current_mol)
                        for idy, reaction in enumerate(reaction_list):
                            current_message += Reactionk.format(i=self.current_state, j=self.current_molid, k=idy)
                            for idz, reactant in enumerate(reaction):
                                if idz > 0:
                                    current_message += ' + '
                                current_message += Singlereactant.format(mol=reactant[0], avail=reactant[1])
                        current_message += ToSelect
                else:
                    raise ValueError(f"Unknown value {value}.")

                # Successful back_state -> back to Phase A.
                self.back_flag = False
                return current_message

            except Exception as e:
                raise ValueError(str(e))

        else:
            raise ValueError(f"Unknown tool {tool_name}.")

    def get_tool_history_context(self) -> str:
        if not self.tool_history:
            return "No tool call history yet."

        context = "Tool call history:\n"
        for i, call in enumerate(self.tool_history):
            context += f"{i+1}. Tool: {call['tool']}\n"
            context += f"   Arguments: {json.dumps(call['args'], ensure_ascii=False)}\n"
            context += f"   Result: {call['result']}\n\n"

        return context

    def get_available_tools_description(self) -> str:
        if not self.tools:
            return "No tools available."

        descriptions = ["Available tools:"]
        for tool in self.tools:
            descriptions.append(tool.get_simple_description())

        return "\n\n".join(descriptions)

    def copy(self, target):
        env = ToolEnvRetroHybrid(
            tools=self.tools,
            max_turns=self.max_turns,
            maxstep=self.maxstep,
            topk=self.topk,
            shuffle=self.shuffle,
            force_noloop=self.force_noloop,
        )
        env.target = target
        env.reset_tracking_variables()
        return env
