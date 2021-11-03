# EMS-COLS-recourse

### Initial Code for [Low-Cost Algorithmic Recourse for Users With Uncertain Cost Functions](https://arxiv.org/abs/2111.01235)

#### Folder structure:

- data folder contains raw and final preprocessed data, along with the pre-processing script.
- Src folder contain the code for our method.
- trained_model contains the trained black box model checkpoint.

  
  

#### Making the environment
```
conda create -n rec_gen python=3.8.1
conda activate rec_gen
pip install -r requirements.txt
```

#### Steps for running experiments.

  

change current working directory to src
```
cd ./src/
```
  

1. Run data_io.py to dump mcmc cost samples.
```
python ./utils/data_io.py --save_data --data_name adult_binary --dump_negative_data --num_mcmc 1000

python ./utils/data_io.py --save_data --data_name compas_binary --dump_negative_data --num_mcmc 1000
```

2. run main experiments on COLS and P-COLS.
```
python run.py --data_name adult_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_main --budget 5000
python run.py --data_name compas_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_main --budget 5000

python run.py --data_name adult_binary --num_mcmc 1000 --model pls --num_cfs 10 --project_name exp_main --budget 5000
python run.py --data_name compas_binary --num_mcmc 1000 --model pls --num_cfs 10 --project_name exp_main --budget 5000
```
  

3. Run ablation Experiments

```
python run.py --data_name adult_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_ablation --budget 3000 --eval cost
python run.py --data_name adult_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_ablation --budget 3000 --eval cost_simple
python run.py --data_name adult_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_ablation --budget 3000 --eval proximity
python run.py --data_name adult_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_ablation --budget 3000 --eval sparsity
python run.py --data_name adult_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_ablation --budget 3000 --eval diversity
```
  

4. Run experiments with budget

```
python run.py --data_name adult_binary --model ls --num_cfs 10 --num_users 100 --project_name exp_budget --budget 500
python run.py --data_name adult_binary --model ls --num_cfs 10 --num_users 100 --project_name exp_budget --budget 1000
python run.py --data_name adult_binary --model ls --num_cfs 10 --num_users 100 --project_name exp_budget --budget 2000
python run.py --data_name adult_binary --model ls --num_cfs 10 --num_users 100 --project_name exp_budget --budget 3000
python run.py --data_name adult_binary --model ls --num_cfs 10 --num_users 100 --project_name exp_budget --budget 5000
python run.py --data_name adult_binary --model ls --num_cfs 10 --num_users 100 --project_name exp_budget --budget 10000
```
  

5. Run experiments with number of counterfactuals
```
python run.py --data_name adult_binary --model model_name --num_cfs 1 --num_users 100 --project_name exp_cfs --budget 5000
python run.py --data_name adult_binary --model model_name --num_cfs 2 --num_users 100 --project_name exp_cfs --budget 5000
python run.py --data_name adult_binary --model model_name --num_cfs 3 --num_users 100 --project_name exp_cfs --budget 5000
python run.py --data_name adult_binary --model model_name --num_cfs 5 --num_users 100 --project_name exp_cfs --budget 5000
python run.py --data_name adult_binary --model model_name --num_cfs 10 --num_users 100 --project_name exp_cfs --budget 5000
python run.py --data_name adult_binary --model model_name --num_cfs 20 --num_users 100 --project_name exp_cfs --budget 5000
python run.py --data_name adult_binary --model model_name --num_cfs 30 --num_users 100 --project_name exp_cfs --budget 5000
```
  

6. Experiment with respect to Monte Carlo samples
- Run these commands for different num_mcmc values. Default set to 5 in commands.
```
python ./utils/data_io.py --save_data --data_name adult_binary --dump_negative_data --num_mcmc 5

python run.py --data_name adult_binary --num_mcmc 5 --model model_name --num_cfs 10 --project_name exp_mcmc --budget 5000 --num_users 100
```  
  
  

#### To train a new blackbox model
- Run this right after preprocessing the data.
```
python train_model.py --data_name adult --max_epochs 1000 --check_val_every_n_epoch=1 --learning_rate=0.0001
```