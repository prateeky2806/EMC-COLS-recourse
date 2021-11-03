'''
General Steps for running experiments.
1. Run process.py in the data folder to dump basic data with train val test split.
    python process.py --data_name adult compas adult_binary gmc_binary compas_binary

2. Run run.py to dump new data with cost samples.
    python run.py --model dice --num_mcmc 500 --alpha 0.5 --save_data --data_name adult
    set the default data: mv dataset_name dataset_name_default

3. Train models.
    python train_model.py --data_name adult --max_epochs 1000 --check_val_every_n_epoch=1 --learning_rate=0.0001
    set the default model: mv model_name model_name_default
4. run experiments.
'''

## Dump datasets.
python process.py --data_name adult compas adult_binary gmc_binary compas_binary

### Training models
python train_model.py --data_name adult_binary --max_epochs 1000 --check_val_every_n_epoch=1 --learning_rate=0.0001
python train_model.py --data_name adult --max_epochs 1000 --check_val_every_n_epoch=1 --learning_rate=0.0001
python train_model.py --data_name compas_binary --max_epochs 2000 --check_val_every_n_epoch=1 --learning_rate=0.0001
python train_model.py --data_name compas --max_epochs 2000 --check_val_every_n_epoch=1 --learning_rate=0.0001
python train_model.py --data_name gmc_binary --max_epochs 1000 --check_val_every_n_epoch=1 --learning_rate=0.0001

## Dump cost functions.
python ./utils/data_io.py --save_data --data_name adult --dump_negative_data --alpha_data --num_mcmc 51
python ./utils/data_io.py --save_data --data_name adult_binary --dump_negative_data --alpha_data --num_mcmc 51
python ./utils/data_io.py --save_data --data_name compas --dump_negative_data --alpha_data --num_mcmc 51
python ./utils/data_io.py --save_data --data_name compas_binary --dump_negative_data --alpha_data --num_mcmc 51


## Comparision across models across datasets  (~9 hours ls amn pls)
python run.py --data_name adult_binary --num_mcmc 1000 --model dice --num_cfs 10 --project_name exp_user40 --budget 5000
python run.py --data_name adult_binary --num_mcmc 1000 --model lime_dice --num_cfs 10 --project_name exp_user40 --budget 5000
python run.py --data_name adult_binary --num_mcmc 1000 --model random --num_cfs 10 --project_name exp_user40 --budget 5000
python run.py --data_name adult_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_user40 --budget 5000
python run.py --data_name adult_binary --num_mcmc 1000 --model pls --num_cfs 10 --project_name exp_user40 --budget 5000 --num_parallel_runs 5
python ./CARLA/carla.py --data_name adult_binary --num_mcmc 1000 --num_cfs 10 --model ar --project_name exp_user40 --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_mcmc 1000 --num_cfs 10 --model face_epsilon --project_name exp_user40 --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_mcmc 1000 --num_cfs 10 --model face_knn --project_name exp_user40 --budget 5000

python run.py --data_name compas_binary --num_mcmc 1000 --model dice --num_cfs 10 --project_name exp_user40 --budget 5000
python run.py --data_name compas_binary --num_mcmc 1000 --model lime_dice --num_cfs 10 --project_name exp_user40 --budget 5000
python run.py --data_name compas_binary --num_mcmc 1000 --model random --num_cfs 10 --project_name exp_user40 --budget 5000
python run.py --data_name compas_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_user40 --budget 5000
python run.py --data_name compas_binary --num_mcmc 1000 --model pls --num_cfs 10 --project_name exp_user40 --budget 5000 --num_parallel_runs 5
python ./CARLA/carla.py --data_name compas_binary --num_mcmc 1000 --num_cfs 10 --model ar --project_name exp_user40 --budget 5000
python ./CARLA/carla.py --data_name compas_binary --num_mcmc 1000 --num_cfs 10 --model face_epsilon --project_name exp_user40 --budget 5000
python ./CARLA/carla.py --data_name compas_binary --num_mcmc 1000 --num_cfs 10 --model face_knn --project_name exp_user40 --budget 5000

python run.py --data_name adult --num_mcmc 1000 --model dice --num_cfs 10 --project_name exp_user41 --budget 5000
#python run.py --data_name adult --num_mcmc 1000 --model lime_dice --num_cfs 10 --project_name exp_user41 --budget 5000
python run.py --data_name adult --num_mcmc 1000 --model random --num_cfs 10 --project_name exp_user41 --budget 5000
python run.py --data_name adult --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_user41 --budget 5000
python run.py --data_name adult --num_mcmc 1000 --model pls --num_cfs 10 --project_name exp_user41 --budget 5000 --num_parallel_runs 5

python run.py --data_name compas --num_mcmc 1000 --model dice --num_cfs 10 --project_name exp_user41 --budget 5000
#python run.py --data_name compas --num_mcmc 1000 --model lime_dice --num_cfs 10 --project_name exp_user41 --budget 5000
python run.py --data_name compas --num_mcmc 1000 --model random --num_cfs 10 --project_name exp_user41 --budget 5000
python run.py --data_name compas --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_user41 --budget 5000
python run.py --data_name compas --num_mcmc 1000 --model pls --num_cfs 10 --project_name exp_user41 --budget 5000 --num_parallel_runs 5


### main variance

python run.py --data_name adult_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_user40 --budget 5000 --run_id
python run.py --data_name adult_binary --num_mcmc 1000 --model pls --num_cfs 10 --project_name exp_user40 --budget 5000 --num_parallel_runs 5 --run_id

python run.py --data_name adult_binary --num_mcmc 1000 --model dice --num_cfs 10 --project_name exp_user40 --budget 5000 --run_id
python run.py --data_name adult_binary --num_mcmc 1000 --model lime_dice --num_cfs 10 --project_name exp_user40 --budget 5000 --run_id
python run.py --data_name adult_binary --num_mcmc 1000 --model random --num_cfs 10 --project_name exp_user40 --budget 5000 --run_id
python ./CARLA/carla.py --data_name adult_binary --num_mcmc 1000 --num_cfs 10 --model ar --project_name exp_user40 --budget 5000 --run_id
python ./CARLA/carla.py --data_name adult_binary --num_mcmc 1000 --num_cfs 10 --model face_epsilon --project_name exp_user40 --budget 5000 --run_id
python ./CARLA/carla.py --data_name adult_binary --num_mcmc 1000 --num_cfs 10 --model face_knn --project_name exp_user40 --budget 5000 --run_id

python run.py --data_name compas_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_user40 --budget 5000 --run_id
python run.py --data_name compas_binary --num_mcmc 1000 --model pls --num_cfs 10 --project_name exp_user40 --budget 5000 --num_parallel_runs 5 --run_id

python run.py --data_name compas_binary --num_mcmc 1000 --model dice --num_cfs 10 --project_name exp_user40 --budget 5000 --run_id
python run.py --data_name compas_binary --num_mcmc 1000 --model lime_dice --num_cfs 10 --project_name exp_user40 --budget 5000 --run_id
python run.py --data_name compas_binary --num_mcmc 1000 --model random --num_cfs 10 --project_name exp_user40 --budget 5000 --run_id
python ./CARLA/carla.py --data_name compas_binary --num_mcmc 1000 --num_cfs 10 --model ar --project_name exp_user40 --budget 5000 --run_id
python ./CARLA/carla.py --data_name compas_binary --num_mcmc 1000 --num_cfs 10 --model face_epsilon --project_name exp_user40 --budget 5000 --run_id
python ./CARLA/carla.py --data_name compas_binary --num_mcmc 1000 --num_cfs 10 --model face_knn --project_name exp_user40 --budget 5000 --run_id



# Exp Ablation  (~24 hours)

python run.py --data_name adult_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_user40_ablation --budget 3000 --eval cost
python run.py --data_name adult_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_user40_ablation --budget 3000 --eval cost_simple
python run.py --data_name adult_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_user40_ablation --budget 3000 --eval diversity
python run.py --data_name adult_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_user40_ablation --budget 3000 --eval dirichlet
python run.py --data_name adult_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_user40_ablation --budget 3000 --eval proximity
python run.py --data_name adult_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_user40_ablation --budget 3000 --eval sparsity

python run.py --data_name compas_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_user40_ablation --budget 3000 --eval cost
python run.py --data_name compas_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_user40_ablation --budget 3000 --eval cost_simple
python run.py --data_name compas_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_user40_ablation --budget 3000 --eval diversity
python run.py --data_name compas_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_user40_ablation --budget 3000 --eval dirichlet
python run.py --data_name compas_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_user40_ablation --budget 3000 --eval proximity
python run.py --data_name compas_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_user40_ablation --budget 3000 --eval sparsity


#python run.py --data_name adult_binary --model pls --num_cfs 10 --num_users 20 --project_name exp_user40_ablation --budget 3000 --eval cost
#python run.py --data_name adult_binary --model pls --num_cfs 10 --num_users 20 --project_name exp_user40_ablation --budget 3000 --eval cost_simple
#python run.py --data_name adult_binary --model pls --num_cfs 10 --num_users 20 --project_name exp_user40_ablation --budget 3000 --eval diversity
#python run.py --data_name adult_binary --model pls --num_cfs 10 --num_users 20 --project_name exp_user40_ablation --budget 3000 --eval dirichlet
#python run.py --data_name adult_binary --model pls --num_cfs 10 --num_users 20 --project_name exp_user40_ablation --budget 3000 --eval proximity
#python run.py --data_name adult_binary --model pls --num_cfs 10 --num_users 20 --project_name exp_user40_ablation --budget 3000 --eval sparsity


## DS train test alpha. alphagrid (~5 hours)

python ./utils/data_io.py --save_data --data_name adult_binary --dump_negative_data --alpha_data --num_mcmc 30
python run.py --data_name adult_binary --model ls --num_users 50 --num_cfs 10 --project_name exp_user41_alphagrid --num_mcmc 30 --budget 3000 --eval cost --alpha 0
python run.py --data_name adult_binary --model ls --num_users 50 --num_cfs 10 --project_name exp_user41_alphagrid --num_mcmc 30 --budget 3000 --eval cost --alpha 0.2
python run.py --data_name adult_binary --model ls --num_users 50 --num_cfs 10 --project_name exp_user41_alphagrid --num_mcmc 30 --budget 3000 --eval cost --alpha 0.4
python run.py --data_name adult_binary --model ls --num_users 50 --num_cfs 10 --project_name exp_user41_alphagrid --num_mcmc 30 --budget 3000 --eval cost --alpha 0.6
python run.py --data_name adult_binary --model ls --num_users 50 --num_cfs 10 --project_name exp_user41_alphagrid --num_mcmc 30 --budget 3000 --eval cost --alpha 0.8
python run.py --data_name adult_binary --model ls --num_users 50 --num_cfs 10 --project_name exp_user41_alphagrid --num_mcmc 30 --budget 3000 --eval cost --alpha 1
python alphagrid_eval.py --project_name exp_user41_alphagrid --num_user 50 --num_mcmc 30 --budget 3000

### DS  test alpha grid. (~15 minutes)
python alphagrid_eval.py --project_name exp_user41_alphagrid1 --num_mcmc 51 --budget 3000


### DS dump (~15 minutes)
python ds_dump.py --num_samples 500 --num_mcmc 25 --data_name adult_binary --subsample --conc_type uniform
python ds_dump.py --num_samples 500 --num_mcmc 25 --data_name adult_binary --subsample --conc_type gamma
python ds_dump.py --num_samples 500 --num_mcmc 25 --data_name adult_binary --conc_type uniform
python ds_dump.py --num_samples 500 --num_mcmc 25 --data_name adult_binary --conc_type gamma

#python ds_dump.py --num_samples 500 --num_mcmc 51 --data_name adult_binary --subsample --conc_type uniform
#python ds_dump.py --num_samples 500 --num_mcmc 51 --data_name adult_binary --subsample --conc_type gamma
#python ds_dump.py --num_samples 500 --num_mcmc 51 --data_name adult_binary --conc_type uniform
#python ds_dump.py --num_samples 500 --num_mcmc 51 --data_name adult_binary --conc_type gamma

#python ds_dump.py --num_samples 500 --num_mcmc 50 --data_name compas_binary --subsample --conc_type uniform
#python ds_dump.py --num_samples 500 --num_mcmc 50 --data_name compas_binary --subsample --conc_type gamma
#python ds_dump.py --num_samples 500 --num_mcmc 51 --data_name compas_binary --subsample --conc_type uniform
#python ds_dump.py --num_samples 500 --num_mcmc 51 --data_name compas_binary --subsample --conc_type gamma
#
#python ds_dump.py --num_samples 500 --num_mcmc 50 --data_name compas_binary --conc_type uniform
#python ds_dump.py --num_samples 500 --num_mcmc 50 --data_name compas_binary --conc_type gamma
#python ds_dump.py --num_samples 500 --num_mcmc 51 --data_name compas_binary --conc_type uniform
#python ds_dump.py --num_samples 500 --num_mcmc 51 --data_name compas_binary --conc_type gamma

## DS EVAL (~10 minutes)

python ds_eval.py --project_name exp_user40 --budget 5000 --num_samples 500 --num_mcmc 25 --data_name adult_binary --subsample --conc_type uniform
python ds_eval.py --project_name exp_user40 --budget 5000 --num_samples 500 --num_mcmc 25 --data_name adult_binary --subsample --conc_type gamma
python ds_eval.py --project_name exp_user40 --budget 5000 --num_samples 500 --num_mcmc 25 --data_name adult_binary --conc_type uniform
python ds_eval.py --project_name exp_user40 --budget 5000 --num_samples 500 --num_mcmc 25 --data_name adult_binary --conc_type gamma

#python ds_eval.py --project_name exp_user40 --budget 5000 --num_samples 500 --num_mcmc 51 --data_name adult_binary --subsample --conc_type uniform
#python ds_eval.py --project_name exp_user40 --budget 5000 --num_samples 500 --num_mcmc 51 --data_name adult_binary --subsample --conc_type gamma
#python ds_eval.py --project_name exp_user40 --budget 5000 --num_samples 500 --num_mcmc 51 --data_name adult_binary --conc_type uniform
#python ds_eval.py --project_name exp_user40 --budget 5000 --num_samples 500 --num_mcmc 51 --data_name adult_binary --conc_type gamma

#python ds_eval.py --project_name exp_user40 --budget 5000 --num_samples 500 --num_mcmc 50 --data_name compas_binary --subsample --conc_type uniform
#python ds_eval.py --project_name exp_user40 --budget 5000 --num_samples 500 --num_mcmc 50 --data_name compas_binary --subsample --conc_type gamma
#python ds_eval.py --project_name exp_user40 --budget 5000 --num_samples 500 --num_mcmc 51 --data_name compas_binary --subsample --conc_type uniform
#python ds_eval.py --project_name exp_user40 --budget 5000 --num_samples 500 --num_mcmc 51 --data_name compas_binary --subsample --conc_type gamma
#
#python ds_eval.py --project_name exp_user40 --budget 5000 --num_samples 500 --num_mcmc 50 --data_name compas_binary --conc_type uniform
#python ds_eval.py --project_name exp_user40 --budget 5000 --num_samples 500 --num_mcmc 50 --data_name compas_binary --conc_type gamma
#python ds_eval.py --project_name exp_user40 --budget 5000 --num_samples 500 --num_mcmc 51 --data_name compas_binary --conc_type uniform
#python ds_eval.py --project_name exp_user40 --budget 5000 --num_samples 500 --num_mcmc 51 --data_name compas_binary --conc_type gamma


# Experiment wrt budget
python run.py --data_name adult_binary --model dice --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 500
python run.py --data_name adult_binary --model dice --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 1000
python run.py --data_name adult_binary --model dice --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 2000
python run.py --data_name adult_binary --model dice --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 3000
python run.py --data_name adult_binary --model dice --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 5000
python run.py --data_name adult_binary --model dice --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 10000

python run.py --data_name adult_binary --model lime_dice --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 500
python run.py --data_name adult_binary --model lime_dice --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 1000
python run.py --data_name adult_binary --model lime_dice --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 2000
python run.py --data_name adult_binary --model lime_dice --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 3000
python run.py --data_name adult_binary --model lime_dice --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 5000
python run.py --data_name adult_binary --model lime_dice --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 10000

python run.py --data_name adult_binary --model random --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 500
python run.py --data_name adult_binary --model random --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 1000
python run.py --data_name adult_binary --model random --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 2000
python run.py --data_name adult_binary --model random --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 3000
python run.py --data_name adult_binary --model random --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 5000
python run.py --data_name adult_binary --model random --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 10000

python run.py --data_name adult_binary --model ls --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 500
python run.py --data_name adult_binary --model ls --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 1000
python run.py --data_name adult_binary --model ls --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 2000
python run.py --data_name adult_binary --model ls --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 3000
python run.py --data_name adult_binary --model ls --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 5000
python run.py --data_name adult_binary --model ls --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 10000

python run.py --data_name adult_binary --model pls --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 500 --num_parallel_runs 5
python run.py --data_name adult_binary --model pls --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 1000 --num_parallel_runs 5
python run.py --data_name adult_binary --model pls --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 2000 --num_parallel_runs 5
python run.py --data_name adult_binary --model pls --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 3000 --num_parallel_runs 5
python run.py --data_name adult_binary --model pls --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 5000 --num_parallel_runs 5
python run.py --data_name adult_binary --model pls --num_cfs 10 --num_users 100 --project_name exp_user40_budget --budget 10000 --num_parallel_runs 5

python ./CARLA/carla.py --data_name adult_binary --num_cfs 10 --num_users 100 --model ar --project_name exp_user40_budget --budget 500
python ./CARLA/carla.py --data_name adult_binary --num_cfs 10 --num_users 100 --model ar --project_name exp_user40_budget --budget 1000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 10 --num_users 100 --model ar --project_name exp_user40_budget --budget 2000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 10 --num_users 100 --model ar --project_name exp_user40_budget --budget 3000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 10 --num_users 100 --model ar --project_name exp_user40_budget --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 10 --num_users 100 --model ar --project_name exp_user40_budget --budget 10000

python ./CARLA/carla.py --data_name adult_binary --num_cfs 10 --num_users 100 --model face_epsilon --project_name exp_user40_budget --budget 500
python ./CARLA/carla.py --data_name adult_binary --num_cfs 10 --num_users 100 --model face_epsilon --project_name exp_user40_budget --budget 1000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 10 --num_users 100 --model face_epsilon --project_name exp_user40_budget --budget 2000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 10 --num_users 100 --model face_epsilon --project_name exp_user40_budget --budget 3000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 10 --num_users 100 --model face_epsilon --project_name exp_user40_budget --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 10 --num_users 100 --model face_epsilon --project_name exp_user40_budget --budget 10000

python ./CARLA/carla.py --data_name adult_binary --num_cfs 10 --num_users 100 --model face_knn --project_name exp_user40_budget --budget 500
python ./CARLA/carla.py --data_name adult_binary --num_cfs 10 --num_users 100 --model face_knn --project_name exp_user40_budget --budget 1000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 10 --num_users 100 --model face_knn --project_name exp_user40_budget --budget 2000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 10 --num_users 100 --model face_knn --project_name exp_user40_budget --budget 3000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 10 --num_users 100 --model face_knn --project_name exp_user40_budget --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 10 --num_users 100 --model face_knn --project_name exp_user40_budget --budget 10000




# Exp num_cfs vs Metrics

python run.py --data_name adult_binary --model dice --num_cfs 1 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model dice --num_cfs 2 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model dice --num_cfs 3 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model dice --num_cfs 5 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model dice --num_cfs 10 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model dice --num_cfs 20 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model dice --num_cfs 30 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model dice --num_cfs 40 --num_users 100 --project_name exp_user40_cfs --budget 5000

python run.py --data_name adult_binary --model lime_dice --num_cfs 1 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model lime_dice --num_cfs 2 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model lime_dice --num_cfs 3 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model lime_dice --num_cfs 5 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model lime_dice --num_cfs 10 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model lime_dice --num_cfs 20 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model lime_dice --num_cfs 30 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model lime_dice --num_cfs 40 --num_users 100 --project_name exp_user40_cfs --budget 5000

python run.py --data_name adult_binary --model random --num_cfs 1 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model random --num_cfs 2 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model random --num_cfs 3 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model random --num_cfs 5 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model random --num_cfs 10 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model random --num_cfs 20 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model random --num_cfs 30 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model random --num_cfs 40 --num_users 100 --project_name exp_user40_cfs --budget 5000

python run.py --data_name adult_binary --model ls --num_cfs 1 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model ls --num_cfs 2 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model ls --num_cfs 3 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model ls --num_cfs 5 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model ls --num_cfs 10 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model ls --num_cfs 20 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model ls --num_cfs 30 --num_users 100 --project_name exp_user40_cfs --budget 5000
python run.py --data_name adult_binary --model ls --num_cfs 40 --num_users 100 --project_name exp_user40_cfs --budget 5000

python run.py --data_name adult_binary --model pls --num_cfs 1 --num_users 100 --project_name exp_user40_cfs --budget 5000 --num_parallel_runs 5
python run.py --data_name adult_binary --model pls --num_cfs 2 --num_users 100 --project_name exp_user40_cfs --budget 5000 --num_parallel_runs 5
python run.py --data_name adult_binary --model pls --num_cfs 3 --num_users 100 --project_name exp_user40_cfs --budget 5000 --num_parallel_runs 5
python run.py --data_name adult_binary --model pls --num_cfs 5 --num_users 100 --project_name exp_user40_cfs --budget 5000 --num_parallel_runs 5
python run.py --data_name adult_binary --model pls --num_cfs 10 --num_users 100 --project_name exp_user40_cfs --budget 5000 --num_parallel_runs 5
python run.py --data_name adult_binary --model pls --num_cfs 20 --num_users 100 --project_name exp_user40_cfs --budget 5000 --num_parallel_runs 5
python run.py --data_name adult_binary --model pls --num_cfs 30 --num_users 100 --project_name exp_user40_cfs --budget 5000 --num_parallel_runs 5
python run.py --data_name adult_binary --model pls --num_cfs 40 --num_users 100 --project_name exp_user40_cfs --budget 5000 --num_parallel_runs 5

python ./CARLA/carla.py --data_name adult_binary --num_cfs 1 --num_users 100 --model face_knn --project_name exp_user40_cfs --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 2 --num_users 100 --model face_knn --project_name exp_user40_cfs --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 3 --num_users 100 --model face_knn --project_name exp_user40_cfs --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 5 --num_users 100 --model face_knn --project_name exp_user40_cfs --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 10 --num_users 100 --model face_knn --project_name exp_user40_cfs --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 20 --num_users 100 --model face_knn --project_name exp_user40_cfs --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 30 --num_users 100 --model face_knn --project_name exp_user40_cfs --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 40 --num_users 100 --model face_knn --project_name exp_user40_cfs --budget 5000

python ./CARLA/carla.py --data_name adult_binary --num_cfs 1 --num_users 100 --model face_epsilon --project_name exp_user40_cfs --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 2 --num_users 100 --model face_epsilon --project_name exp_user40_cfs --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 3 --num_users 100 --model face_epsilon --project_name exp_user40_cfs --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 5 --num_users 100 --model face_epsilon --project_name exp_user40_cfs --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 10 --num_users 100 --model face_epsilon --project_name exp_user40_cfs --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 20 --num_users 100 --model face_epsilon --project_name exp_user40_cfs --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 30 --num_users 100 --model face_epsilon --project_name exp_user40_cfs --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 40 --num_users 100 --model face_epsilon --project_name exp_user40_cfs --budget 5000

python ./CARLA/carla.py --data_name adult_binary --num_cfs 1 --num_users 100 --model ar --project_name exp_user40_cfs --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 2 --num_users 100 --model ar --project_name exp_user40_cfs --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 3 --num_users 100 --model ar --project_name exp_user40_cfs --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 5 --num_users 100 --model ar --project_name exp_user40_cfs --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 10 --num_users 100 --model ar --project_name exp_user40_cfs --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 20 --num_users 100 --model ar --project_name exp_user40_cfs --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 30 --num_users 100 --model ar --project_name exp_user40_cfs --budget 5000
python ./CARLA/carla.py --data_name adult_binary --num_cfs 40 --num_users 100 --model ar --project_name exp_user40_cfs --budget 5000



##### MCMC experiment

## Data
python ./utils/data_io.py --save_data --data_name adult_binary --dump_negative_data --num_mcmc 1
python run.py --data_name adult_binary --num_mcmc 1 --model ls --num_cfs 10 --project_name exp_user40_mcmc2 --budget 5000 --num_users 100

python ./utils/data_io.py --save_data --data_name adult_binary --dump_negative_data --num_mcmc 5
python run.py --data_name adult_binary --num_mcmc 5 --model ls --num_cfs 10 --project_name exp_user40_mcmc2 --budget 5000 --num_users 100

python ./utils/data_io.py --save_data --data_name adult_binary --dump_negative_data --num_mcmc 10
python run.py --data_name adult_binary --num_mcmc 10 --model ls --num_cfs 10 --project_name exp_user40_mcmc2 --budget 5000 --num_users 100

python ./utils/data_io.py --save_data --data_name adult_binary --dump_negative_data --num_mcmc 20
python run.py --data_name adult_binary --num_mcmc 20 --model ls --num_cfs 10 --project_name exp_user40_mcmc2 --budget 5000 --num_users 100

python ./utils/data_io.py --save_data --data_name adult_binary --dump_negative_data --num_mcmc 30
python run.py --data_name adult_binary --num_mcmc 30 --model ls --num_cfs 10 --project_name exp_user40_mcmc2 --budget 5000 --num_users 100

python ./utils/data_io.py --save_data --data_name adult_binary --dump_negative_data --num_mcmc 100
python run.py --data_name adult_binary --num_mcmc 100 --model ls --num_cfs 10 --project_name exp_user40_mcmc2 --budget 5000 --num_users 100

python ./utils/data_io.py --save_data --data_name adult_binary --dump_negative_data --num_mcmc 200
python run.py --data_name adult_binary --num_mcmc 200 --model ls --num_cfs 10 --project_name exp_user40_mcmc2 --budget 5000 --num_users 100

python ./utils/data_io.py --save_data --data_name adult_binary --dump_negative_data --num_mcmc 300
python run.py --data_name adult_binary --num_mcmc 300 --model ls --num_cfs 10 --project_name exp_user40_mcmc2 --budget 5000 --num_users 100

python ./utils/data_io.py --save_data --data_name adult_binary --dump_negative_data --num_mcmc 500
python run.py --data_name adult_binary --num_mcmc 500 --model ls --num_cfs 10 --project_name exp_user40_mcmc2 --budget 5000 --num_users 100

python ./utils/data_io.py --save_data --data_name adult_binary --dump_negative_data --num_mcmc 1000
python run.py --data_name adult_binary --num_mcmc 1000 --model ls --num_cfs 10 --project_name exp_user40_mcmc2 --budget 5000 --num_users 100

python ./utils/data_io.py --save_data --data_name adult_binary --dump_negative_data --num_mcmc 2000
python run.py --data_name adult_binary --num_mcmc 2000 --model ls --num_cfs 10 --project_name exp_user40_mcmc2 --budget 5000 --num_users 100



