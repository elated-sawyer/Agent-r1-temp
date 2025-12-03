"""
Single step retro tool implementation
"""

from typing import Dict
import json
from agent_r1.tool.tool_base import Tool
from mlp_retrosyn.mlp_inference import MLPModel
import pickle
import pandas as pd
import random

class SingleStepRetroTool(Tool):
    """
    Tool for performing Single step retro
    """
    
    def __init__(self, mlp_model_dump='./one_step_model/saved_rollout_state_1_2048.ckpt'):
        """
        Initialize the calculator tool
        """
        name = "single_step_retro"
        description = "Perform single step retrosynthesis for a molecule and return several possible reactions to synthesize it. Note that the reactions may be incorrect and each reactant is marked as 'available' or 'unavailable'. The unavailable molecules have to be synthesized further."
        parameters = {
            "type": "object",
            "properties": {
                "molecule": {
                    "type": "string",
                    "description": "The ID for the molecule to be synthesized. For example, 0-0 ."
                }
            },
            "required": ["molecule"]
        }
        dirpath = '.'
        mlp_templates=dirpath+'/one_step_model/template_rules_1.dat'
        if mlp_model_dump == './one_step_model/saved_rollout_state_1_2048.ckpt':
            self.use_default_mlp = True
        else:
            self.use_default_mlp = False
        
        self.one_step = MLPModel(mlp_model_dump, mlp_templates, device=-1)

        self.mol_info = []
        starting_molecules=dirpath+'/dataset/origin_dict_canonical.csv'
        if starting_molecules[-3:] == 'csv':
            self.starting_mols = set(list(pd.read_csv(starting_molecules)['mol']))
        else:
            assert starting_molecules[-3:] == 'pkl'
            with open(starting_molecules, 'rb') as f:
                starting_mols = pickle.load(f)
                self.starting_mols = set(list(starting_mols))
        
        super().__init__(name, description, parameters)
    
    def execute(self, args: Dict) -> str:
        """
        Execute calculator operations
        
        Args:
            args: Tool parameters, containing:
                - "expression": arithmetic expression to evaluate
            
        Returns:
            Result of the calculation
        """
        mol = args.get("molecule", "").strip()
        topk = args.get("topk", 5)
        shuffle = args.get("shuffle", False)
        reaction_list = args.get("reaction_list", [])
        exist = args.get("exist", False)

        if exist:
            return {"results": reaction_list}

        
        # in fact, this won't happen
        if not mol:
            return {"error": "No molecule provided."}
        
        try:
            if mol in self.mol_info:
                assert not self.mol_info[mol]['exist'], f"This molecule {mol} is already solved."
                reactants = self.mol_info[mol]['reactants']
            else:
                assert mol not in self.starting_mols, f"This molecule {mol} is already solved."
                if self.use_default_mlp:
                    results = self.one_step.run(mol, topk=50)
                else:
                    results = self.one_step.run(mol, topk=topk)
                if not results:
                    return {"results": []}
                reactants_ = results['reactants']
                scores = results['scores']
                reactants = []
                for j in range(len(scores)):
                    # reactant_list = list(set(reactants[j].split('.'))) ['a', 'b'] [['a', True], ['b', False]]
                    reactant_list = reactants_[j].split('.')
                    reactant_list_exist = []
                    for reactant in reactant_list:
                        if reactant in self.starting_mols:
                            reactant_list_exist.append([reactant, True])
                        else:
                            reactant_list_exist.append([reactant, False])
                    reactants.append([reactant_list_exist, reactants_[j], scores[j]])
            
            reactants_sorted = sorted(reactants, key=lambda x: x[-1], reverse=True)
            reactants_sorted_used = reactants_sorted[:topk]
            reactants_sorted_used_list = []
            for reaction in reactants_sorted_used:
                ll = []
                for r in reaction[0]:
                    if r[1]:
                        ll.append([r[0], 'available'])
                    else:
                        ll.append([r[0], 'unavailable'])
                reactants_sorted_used_list.append(ll)
            if shuffle:
                random.shuffle(reactants_sorted_used_list)

            result = {"results": reactants_sorted_used_list}
            
            return result
        
        except Exception as e:
            return {"error": str(e)}
    
    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        TODO: implement the reward for molecule selection
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value
        """
        try:
            # result_obj = json.loads(result)
            result_obj = result
            # 有效的工具调用
            if "results" in result_obj:
                return 0.0
            elif "error" in result_obj:
                return 0.0 # -0.1  # 轻微惩罚错误
            else:
                return 0.0
        except:
            return 0.0 #-0.1  # 无法解析结果

    
if __name__ == "__main__":
    calculator_tool = SingleStepRetroTool()
    print(calculator_tool.execute({"expression": "2 + 3"}))
    print(calculator_tool.calculate_reward({"expression": "2 + 3"}, "Result: 5"))
