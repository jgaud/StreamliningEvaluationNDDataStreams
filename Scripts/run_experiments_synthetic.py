from util_functions import *
from sklearn.model_selection import ParameterGrid
from itertools import repeat
from pathos.pools import ProcessPool

param_grid = {
    'wait_time_between_classes': [0, 5000, 20000],
    'sparsity': [1],
    'ratio_offline': [0.1, 0.4, 0.7],
    'ratio_known_classes': [0.3, 0.5, 0.7],
    'random_state': [42],
    }

h_params = {
        'kini': 10,
        'cluster_algorithm': 'kmeans', 
        'window_size': 3000, 
        'threshold_strategy': 1, 
        'threshold_factor': 1.1, 
        'min_short_mem_trigger': 500, 
        'min_examples_cluster': 50,
        'random_state': 42,
        'update_summary': False,
        'verbose': 0,
        }

filename = sys.argv[1]
n_samples = int(sys.argv[2])
target = sys.argv[3]
n_jobs = int(sys.argv[4])
sl = slice(*map(lambda x: int(x.strip()) if x.strip() else None, sys.argv[5].split(':')))
aim_dir = sys.argv[6]

print(f'Loading dataset {filename}...', flush=True)
if filename.endswith('.arff'):
    df = load_MOA(f'/home/jgaud/projects/def-pbranco/jgaud/NoveltyDetectionMetrics/Data/{filename}')
elif filename.endswith('.csv'):
    df = load_CSV(f'/home/jgaud/projects/def-pbranco/jgaud/NoveltyDetectionMetrics/Data/{filename}')
else:
    print('File extension not supported', flush=True)

pool = ProcessPool(n_jobs)

pool.map(start_experiment, list(ParameterGrid(param_grid))[sl], repeat(target), repeat(df), repeat(h_params), repeat(n_samples), repeat(filename), repeat(aim_dir))

pool.close()
pool.join()