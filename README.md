# Retro-R1 for Retrosynthetic Planning

This repository is based on the open-source codebase of [Retro*](https://github.com/binghong-ml/retro_star) and [Agent-R1](https://github.com/0russwest0/Agent-R1).

## Setup

### Create conda environment
```bash
cd Retro-R1
conda env create --file environment.yml
conda activate Retro_R1
pip install flash-attn==2.7.4.post1
pip install -e packages/mlp_retrosyn
pip install -e packages/rdchiral
unzip verl.zip
cd verl
pip install --no-deps -e .
```

### Download the necessary files
To reproduce the results in the paper, we also need the additional files containing the training dataset, evaluation datasets (USPTO, ChEMBL-1000), starting molecules, and the template rules.
Files from Retro* (such as single-step model weights V1) can be downloaded from the [link](https://www.dropbox.com/s/ar9cupb18hv96gj/retro_data.zip?dl=0), and put the folders (```dataset/```, ```one_step_model/```) under this directory. The single-step model weights V2 and V3 can be downloaded from the [link](https://drive.google.com/drive/u/0/folders/13DdftEV0x55OZ8ZxHNAkmcvi_4x90hPI) and put the files (```retro_star_value_ours.ckpt```(V2) and ```retro_star_zero_ours.ckpt```(V3)) under the ```one_step_model/``` directory. The single-step model weights V4 is not released and can be trained following the instructions in [PDVN](https://github.com/DiXue98/PDVN). We can also provide the trained weights if requested. Put the trained weights under the ```one_step_model/``` directory and name it as ```retro_star_V4.ckpt```. The ChEMBL-1000 testset can be downloaded from [link](https://drive.google.com/drive/folders/198WuPlSyMeMvvd4i2SM833jPAcGllzDu?usp=sharing) and put under the folder ```dataset/```.


## Preprocess the dataset
```bash
python ./examples/data_preprocess/reaction.py
```

## Training
```bash
bash training_script.py
```

Note that this script only use one node with eight gpus. To use multi nodes, please follow the instructions of [verl](https://github.com/volcengine/verl).

## Export model
Modify the scripts ```export.sh``` and change the ori_pth, ckpt_pth and export_pth to export model from the best checkpoint.
```bash
bash export.py
```


## Evaluation on Retro*-190 testset
```bash
bash test_retro_script.py
```

## Evaluation on ChEMBL-1000 testset
```bash
bash test_chembl_script.py
```
