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
Preprocess the HotpotQA dataset to parquet format
"""

import os
import datasets
import argparse
import json
import requests
from tqdm import tqdm
import zipfile
import pickle
import random
from rdkit import Chem
from verl.utils.hdfs_io import copy, makedirs

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

if __name__ == '__main__':
    data_source = 'reaction_pathway_search'
    train_data_path = './dataset/routes_train.pkl'
    test_data_path = './dataset/routes_possible_test_hard.pkl'
    ood_test_data_path = './dataset/chembl_1000.pkl'
    train_size = None
    local_dir = './data/reaction_pathway_search/'
    exclude=4

    with open(train_data_path, 'rb') as f:
        trainset = pickle.load(f)
    
    with open(test_data_path, 'rb') as f:
        testset = pickle.load(f)

    # process train:
    maxstep_ub = 30
    train_question = []
    train_answer = []
    train_level = []
    for iid, route in enumerate(trainset):
        if iid % 1000 == 0:
            print(iid)

        target = route[0].split('>')[0]
        targetmol = canonicalize_smiles_clear_map(target)
        if len(route) == 1 or len(route) > maxstep_ub:
            continue
        if len(route) <= exclude:
            continue
            # train_level.append('easy')
        elif len(route) <= 10:
            train_level.append('middel')
        else:
            train_level.append('hard')

        train_question.append(targetmol)
        train_answer.append(json.dumps(route))
        
    
    train_dataset = datasets.Dataset.from_dict({
        'question': train_question,
        'answer': train_answer,
        'level': train_level
    })

    test_question = []
    test_answer = []
    test_level = []
    for iid, route in enumerate(testset):
        target = route[0].split('>')[0]
        targetmol = canonicalize_smiles_clear_map(target)
        test_question.append(targetmol)
        test_answer.append(json.dumps(route))
        test_level.append('hard')
    
    test_dataset = datasets.Dataset.from_dict({
        'question': test_question,
        'answer': test_answer,
        'level': test_level
    })

    
    with open(ood_test_data_path, 'rb') as f:
        ood_testset = pickle.load(f)

    ood_test_question = []
    ood_test_answer = []
    ood_test_level = []
    for iid, route in enumerate(ood_testset):
        target = route[0].split('>')[0]
        targetmol = canonicalize_smiles_clear_map(target)
        ood_test_question.append(targetmol)
        ood_test_answer.append(json.dumps(route))
        ood_test_level.append('hard')
    
    ood_test_dataset = datasets.Dataset.from_dict({
        'question': ood_test_question,
        'answer': ood_test_answer,
        'level': ood_test_level
    })


    if train_size is not None:
        indices = random.sample(range(len(train_dataset)), train_size)
        train_dataset = train_dataset.select(indices)

    instruction_following = """You are a professional organic chemist, skilled at finding pathways to synthesize novel molecules. Given a target molecule, your task is to find a pathway where the target can be eventually synthesized using ingredient molecules, and each step of the pathway is a viable chemical reaction. The maximum number of reaction steps you can use is {n}.
You can use the tools provided to you to find the pathway. You can use the tool as many times as you want.
You must first conduct reasoning inside <think>...</think>. Then use the tool call <tool_call>...</tool_call> to call the tool.

Output format for tool call:
<think>
...
</think>
<tool_call>
...
</tool_call>

"""
    question_following = """The target molecule is {mol}. Now start to search for the pathway.
Molecule state 0: Unsolved molecules:
Molecule 0-0: {mol}
Now, 0 steps are used.
Please select the molecule with the highest synthesis difficulty from the unsolved molecules. Use the tool 'single_step_retro' to synthesize it."""          

    maxstep = 30
    # Process each data item
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop('question')
            question = instruction_following.format(n=maxstep) + question_following.format(mol=question_raw)
            
            answer_raw = example.pop('answer')
            
            # Parse the supporting facts from JSON string back to Python object if needed
            # supporting_facts_str = example.get('supporting_facts', '[]')
            # try:
            #     supporting_facts = json.loads(supporting_facts_str)
            # except (json.JSONDecodeError, TypeError):
            #     supporting_facts = []
            
            # Convert all data to string format to avoid type issues
            data = {
                "target": question_raw,
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer_raw
                },
                "extra_info": {
                    'split': split,
                    'index': str(idx),
                    'answer': answer_raw,
                    'question': question_raw,
                    'level': str(example.get('level', '')),
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    validation_dataset = test_dataset.map(function=make_map_fn('validation'), with_indices=True)
    ood_test_dataset = ood_test_dataset.map(function=make_map_fn('validation'), with_indices=True)

    # print(len(train_dataset))
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train_h4_10.parquet'))
    validation_dataset.to_parquet(os.path.join(local_dir, 'validation_retro_190.parquet'))
    ood_test_dataset.to_parquet(os.path.join(local_dir, 'validation_chembl_1000.parquet'))
