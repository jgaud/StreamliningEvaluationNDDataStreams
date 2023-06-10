import sys
sys.path.append('/home/jgaud/projects/def-pbranco/jgaud/NoveltyDetectionMetrics/streamndr-private')
from streamndr.model import Minas, ECSMinerWF

import math
import numpy as np
from aim import Run, Distribution

from streamndr.metrics import ConfusionMatrixNovelty, MNew, FNew, ErrRate, CER, TTD, TTC
from river.metrics import MacroF1, Rand, AdjustedRand, MutualInfo, AdjustedMutualInfo, VBeta, FowlkesMallows
from river import preprocessing
from river.utils import numpy2dict
import time
from scipy.io import arff
import pandas as pd
import numpy as np

from collections import deque
import math


def online_training(clf, X_test, y_test, known_classes, scaler, aim_run=None, per_class_metrics=True, log_every=1):
    i = 0
    y_preds = []

    conf_matrix = ConfusionMatrixNovelty(known_classes)
    cer = CER(known_classes, cm=conf_matrix)
    m_new = MNew(known_classes, cm=conf_matrix)
    f_new = FNew(known_classes, cm=conf_matrix)
    err_rate = ErrRate(known_classes, cm=conf_matrix)
    ttd = TTD(known_classes)
    ttc = TTC(known_classes)
    rand_index = Rand(cm=conf_matrix)
    adj_rand_index = AdjustedRand(cm=conf_matrix)
    adj_mut_info = AdjustedMutualInfo(cm=conf_matrix)
    mut_info = MutualInfo(cm=conf_matrix)
    v_measure = VBeta(cm=conf_matrix)
    fowlkes_mallows = FowlkesMallows(cm=conf_matrix)
    

    for x, y_true in zip(X_test, y_test):
        start_time = time.time()

        x_dict = numpy2dict(x)
        x = scaler.learn_one(x_dict).transform_one(x_dict)
        
        if isinstance(x, dict):
            y_pred = clf.predict_one(x, y_true) #predict_one takes python dictionaries as per River API
        else:
            y_pred = clf.predict_many(x.reshape(1,-1), [y_true])
            

        if y_pred is not None: #Update our metrics
            y_preds.append(y_pred[0])

            conf_matrix = conf_matrix.update(y_true, y_pred[0])
            ttd = ttd.update(y_true, y_pred[0])
            ttc = ttc.update(y_true, y_pred[0])
            
            if i % log_every == 0:

                try:
                    inv_harm_mean = 2 / ((1/(1-m_new.get())) + (1/(1-f_new.get())))
                except:
                    inv_harm_mean = 0.0

                try:
                    precision = conf_matrix.true_positives_novelty()/(conf_matrix.true_positives_novelty()+conf_matrix.false_positives_novelty())
                except:
                    precision = 0.0

                try:
                    recall = conf_matrix.true_positives_novelty()/(conf_matrix.true_positives_novelty()+conf_matrix.false_negatives_novelty())
                except:
                    recall = 0.0
    
                try:
                    f1s_binary = 2 * (precision * recall) / (precision + recall)
                except:
                    f1s_binary = 0.0

                try:
                    macro_f1 = MacroF1(cm=conf_matrix.get_associated_classes()).get()
                except:
                    macro_f1 = 0.0

                ##Update all metrics to Aim
                if aim_run is not None:
                    metrics_dict = {
                        'time_taken': time.time() - start_time,
                        'inv_m_new': 1-m_new.get(),
                        'inv_f_new': 1-f_new.get(),
                        'inv_harm_mean': inv_harm_mean,
                        'f1_binary': f1s_binary,
                        'macro_f1': macro_f1,
                        'unk_rate': clf.get_unknown_rate(),
                        'err_rate': err_rate.get(),
                        'cer': cer.get(),
                        'aic': cer.get_aic(),
                        'rand_index': rand_index.get(),
                        'adj_rand_index': adj_rand_index.get(),
                        'adj_mut_info': adj_mut_info.get(),
                        'mut_info': mut_info.get(),
                        'v_measure': v_measure.get(),
                        'fowlkes_mallows': fowlkes_mallows.get(),
                    }


                    aim_run.track(metrics_dict, step=i)

                    if per_class_metrics:
                        tmp = clf.get_class_unknown_rate()
                        for key in tmp:
                            aim_run.track(tmp[key], name='class_unk_rate_'+str(key), step=i, context={'metric': 'class_unk_rate', 'class': int(key)})

                        tmp = ttc.get()
                        for key in tmp:
                            aim_run.track(tmp[key], name='ttc_'+str(key), step=i, context={'metric': 'ttc', 'class': int(key)})

        ## Print progress
        i += 1
        if i % 5000 == 0:
                print(i, '/', len(X_test), flush=True)
    
    ## Update metrics on the last step
    if aim_run is not None:
        
        tmp = ttd.get()
        for key in tmp:
            aim_run.track(tmp[key], name='ttd_'+str(key), step=i-1, context={'metric': 'ttd', 'class': int(key)})
            
        d = Distribution(y_preds)
        aim_run.track(d, name='dist', step=i-1, context={'subset': 'y_preds'})
        
    
    return y_preds

def setup_experiment(experiment_name, algorithm, X_train, X_test, y_train, y_test, dataset_params, aim_dir):
    run = Run(experiment=experiment_name, repo=aim_dir, system_tracking_interval=None)
    run.add_tag(algorithm)

    for key in dataset_params:
        run[key] = dataset_params[key]

    d = Distribution(X_train)
    run.track(d, name='dist', step=0, context={'subset': 'X_offline'})
    d = Distribution(X_test)
    run.track(d, name='dist', step=0, context={'subset': 'X_online'})

    d = Distribution(y_train)
    run.track(d, name='dist', step=0, context={'subset': 'y_offline'})
    d = Distribution(y_test)
    run.track(d, name='dist', step=0, context={'subset': 'y_online'})

    return run

def load_MOA(path):
    data = arff.loadarff(path)
    df = pd.DataFrame(data[0])
    
    for i, cl in enumerate(np.unique(df.iloc[:,-1])):
        df.iloc[:,-1].replace(cl, i, inplace=True)
    
    return df

def load_CSV(path):
    return pd.read_csv(path)

def prepare_data(df, wait_time_between_classes, sparsity, offline_size, online_size, known_classes, target, random_state=42):

    # Variables
    current_known_classes = known_classes.copy()

    if random_state is not None:
        np.random.seed(random_state)

    # Move target column to last position
    new_cols = [col for col in df.columns if col != target] + [target]
    df = df[new_cols]
    
    df.reset_index(inplace=True, drop=True) # Reset index to keep track of the order of the samples

    # Create the offline dataframe using known classes
    known_df = df.loc[df[target].isin(known_classes)]
    unknown_df = df.loc[df.index.difference(known_df.index)]

    if math.floor(offline_size) > known_df.shape[0]:
        raise Exception(f"Not enough samples of known classes: {known_df.shape[0]} need {math.floor(offline_size)}")

    offline_df = known_df.iloc[:offline_size, :] # Take offline_size first samples
    rest_df = known_df.iloc[offline_size:, :]
    

    rest_df = pd.concat([rest_df, unknown_df])
    
    rest_df.sort_index(inplace=True) # Reorder the rest of the samples

    # Separate all classes
    all_classes = list(np.unique(df[target]))
    df_per_class = {}

    for cl in all_classes:
        df_per_class[cl] = deque(rest_df.loc[rest_df[target] == cl].values) # https://docs.python.org/3.8/library/collections.html#collections.deque

    # Create the online stream considering the wait time
    online = deque()
    count = 0
    latest_added_nc = -999

    for i in range(online_size):
        if count < wait_time_between_classes:
            cl = get_cl_given_sparsity(current_known_classes, sparsity, latest_added_nc) # Get a random class from all classes given the sparsity

            # Get another random class if we don't have any samples left of the given class
            while len(df_per_class[cl]) == 0 and len(current_known_classes) > 0:
                current_known_classes.remove(cl)
                all_classes.remove(cl)
                if len(current_known_classes) == 0:
                    print("Not enough samples of known classes for the given wait time, skipping...", flush=True)
                    return None, None
                
                cl = get_cl_given_sparsity(current_known_classes, sparsity, latest_added_nc)

            if len(df_per_class[cl]) == 0:
                raise Exception("Not enough samples of known classes for the given wait time.")

            online.append(df_per_class[cl].popleft()) # Get first item from that class and append it to stream
            count += 1

        else:
            cl = get_cl_given_sparsity(all_classes, sparsity, latest_added_nc) # Get a random class from all classes given the sparsity

            # Get another random class if we don't have any samples left of the given class
            while len(df_per_class[cl]) == 0 and len(all_classes) > 0:
                all_classes.remove(cl)
                if len(current_known_classes) == 0:
                    print("Not enough samples, skipping...", flush=True)
                    return None, None
                
                cl = get_cl_given_sparsity(all_classes, sparsity, latest_added_nc)

            if len(df_per_class[cl]) == 0:
                raise Exception("Not enough samples.")

            latest_added_nc = cl
            
            if cl not in current_known_classes:
                current_known_classes.append(cl)
                count = 0
                

            online.append(df_per_class[cl].popleft())

    online_df = pd.DataFrame(online, columns=new_cols) # Transform list to DataFrame
    online_df[target] = online_df[target].astype(int) # Transform label to int

    return offline_df, online_df

def get_cl_given_sparsity(classes, sparsity, latest_added_nc):
    if latest_added_nc in classes:
        weight = 1 / (len(classes)+sparsity-1)
        probs = [weight if elem != latest_added_nc else weight * sparsity for elem in classes]
        return np.random.choice(classes, p=probs)
    
    else:
        return np.random.choice(classes)
    
def start_experiment(param, target, df, h_params, n_samples, filename, aim_dir):
    start_time = time.time()
    
    print(f'Starting experiment with: {param}', flush=True)
    
    param['known_classes'] = np.unique(df[target])[:math.ceil(param['ratio_known_classes']*len(np.unique(df[target])))].tolist()
    offline_df, online_df = prepare_data(df=df, 
                                         wait_time_between_classes=param['wait_time_between_classes'], 
                                         sparsity=param['sparsity'], 
                                         offline_size=math.floor(n_samples*param['ratio_offline']), 
                                         online_size=math.ceil(n_samples*(1-param['ratio_offline'])), 
                                         known_classes=param['known_classes'], 
                                         target=target, 
                                         random_state=param['random_state'])
    if offline_df is None:
        return
    
    X_offline = offline_df.drop(target, axis=1)
    X_online = online_df.drop(target, axis=1).values
    y_offline = offline_df[target]
    y_online = online_df[target]
    
    run = setup_experiment(filename, 'MINAS', 
                           X_offline, X_online, 
                           y_offline, y_online, param, aim_dir)
    run['hparams'] = h_params
    
    minas = Minas(**h_params)
    
    print('Computing offline phase...', flush=True)
    scaler = preprocessing.StandardScaler()
    scaler = scaler.learn_many(X_offline)
    X_offline = scaler.transform_many(X_offline)
    
    minas.learn_many(X_offline, y_offline)
    
    print('Computing online phase...', flush=True)
    online_training(minas, X_online, y_online, param['known_classes'], scaler, aim_run=run, per_class_metrics=True, log_every=100)
    
    run.finalize()
    
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)