# MARL-LLM
This repository contains the code for our paper LAMARL: LLM-Aided Multi-Agent Reinforcement Learning for Cooperative Policy Generation. LAMARL consists of two main components: an LLM-based automatic reward function generation module and a MARL module.

## Requirements
Only Ubuntu 20.04 is recommended.

## Installation
1. Create a new virtual environment using the following command:
```bash
   conda create -n xxx(your env name) python=3.10
```
2. Navigate to the 'marl_llm' folder and run the following command to install dependencies:
```bash
   pip install -r requirements.txt
```
3. Visit the [PyTorch official website](https://pytorch.org/get-started/previous-versions/) and install the GPU version of PyTorch according to your system configuration, such as
```bash
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```
4. Navigate to the 'cus_gym' folder and run the following command to install the MARL-LLM environment:
```bash
   pip install -e .
```
5. If you encounter any other missing packages during the process, feel free to install them manually using ``pip install xxx``

6. The author is using VSCode, so it is recommended to add the following two paths to the bashrc file:
```bash
export PYTHONPATH="$PYTHONPATH:/home/your_project_path/marl_llm/"
```
7. Compile C++ shared library (Some functions in the environment are implemented in C++ to accelerate the sampling process.):
```bash
cd your_path/cus_gym/gym/envs/customized_envs/envs_cplus
chmod +x build.sh
./build.sh
```

8. If you want to use the LLM to generate reward functions, please make sure the API and key are properly configured. Then, run the following script:
```bash
python ./marl_llm/llm/modules/framework/actions/rl_generate_functions.py
```

## Example
### training
1. Set the variable ``image_folder`` in ``cfg/assembly_cfg.py``, for example, to ``'/home/your_path_to_fig/'``
2. Begin training:
```bash
cd your_path/marl_llm/train
python train_assembly.py
```

### evaluation
Copy the name of the experimental directory you ran and replace ‵`curr_run = '2025-01-19-15-58-03'`` in ``eval_assembly.py``, then:
```bash
python eval_assembly.py
```

## Troubleshooting
Please open an [Issue](https://github.com/Guobin-Zhu/MARL-LLM/issues) if you have some trouble and advice.

The document is being continuously updated.
