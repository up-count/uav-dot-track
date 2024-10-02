import sys
from multiprocessing import freeze_support

sys.path.insert(0, './eval/TrackEval/')
import trackeval  # noqa: E402


freeze_support()

# Command line interface:
default_eval_config = trackeval.Evaluator.get_default_eval_config()
default_eval_config['DISPLAY_LESS_PROGRESS'] = False
default_eval_config['PRINT_RESULTS'] = False
default_eval_config['OUTPUT_SUMMARY'] = False
default_eval_config['OUTPUT_EMPTY_CLASSES'] = False
default_eval_config['OUTPUT_DETAILED'] = False
default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.0}
config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs


eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

# dataset_config['CLASSES_TO_EVAL'] = ['porpoise']
dataset_config['SPLIT_TO_EVAL'] = 'test'
dataset_config['DO_PREPROC'] = False
dataset_config['GT_FOLDER'] = './eval/gt/dronecrowd/'
dataset_config['TRACKERS_FOLDER'] = './outputs/preds/dronecrowd/'
dataset_config['SEQMAP_FILE'] = './eval/gt/dronecrowd_testlist.txt'

# Run code
evaluator = trackeval.Evaluator(eval_config)
dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

metrics_list = []
for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
    if metric.get_name() in metrics_config['METRICS']:
        metrics_list.append(metric(metrics_config))
if len(metrics_list) == 0:
    raise Exception('No metrics selected for evaluation')

x = evaluator.evaluate(dataset_list, metrics_list)

stats = {}

for alg in x[0]['MotChallenge2DBox'].keys():
    sequences = x[0]['MotChallenge2DBox'][alg].keys()
    
    metrics = {
        'HOTA': [],
        'MOTA': [],
        'MOTP': [],
        'IDSW': [],
        'IDF1': [],
        'GT_IDs': [],
        'IDs': [],
    }
    
    for seq in sequences:
        if seq == 'COMBINED_SEQ':
            continue
        else:
            for metric_group in x[0]['MotChallenge2DBox'][alg][seq]['pedestrian'].keys():
                for metric in x[0]['MotChallenge2DBox'][alg][seq]['pedestrian'][metric_group].keys():
                    
                    if metric == 'HOTA(0)':
                        metrics['HOTA'] += [x[0]['MotChallenge2DBox'][alg][seq]['pedestrian'][metric_group][metric]]
                    if metric == 'MOTA':
                        metrics['MOTA'] += [x[0]['MotChallenge2DBox'][alg][seq]['pedestrian'][metric_group][metric]]
                    if metric == 'MOTP':
                        metrics['MOTP'] += [x[0]['MotChallenge2DBox'][alg][seq]['pedestrian'][metric_group][metric]]
                    if metric == 'IDSW':
                        metrics['IDSW'] += [x[0]['MotChallenge2DBox'][alg][seq]['pedestrian'][metric_group][metric]]
                    if metric == 'IDF1':
                        metrics['IDF1'] += [x[0]['MotChallenge2DBox'][alg][seq]['pedestrian'][metric_group][metric]]
                    if metric == 'GT_IDs':
                        metrics['GT_IDs'] += [x[0]['MotChallenge2DBox'][alg][seq]['pedestrian'][metric_group][metric]]
                    if metric == 'IDs':
                        metrics['IDs'] += [x[0]['MotChallenge2DBox'][alg][seq]['pedestrian'][metric_group][metric]]
    
    stats[alg] = metrics

# print results as a table, e.g.:

# Alg | HOTA | MOTA | MOTP | IDSW | IDF1 | GT_IDs | IDs
# -------------------------------------------------------
# A   | 0.5  | 0.6  | 0.7  | 0.8  | 0.9  | 10     | 10

for alg, st in stats.items():
    GT_IDS = st['GT_IDs']
    IDs = st['IDs']

    TRCount = []
    TRRelCount = []

    for gtids, ids in zip(GT_IDS, IDs):
        TRCount.append(abs(gtids - ids))
        TRRelCount.append(abs(gtids - ids) / gtids)

    st['TRCount'] = TRCount
    st['TRRelCount'] = TRRelCount

print(f'Alg'.ljust(10), end=' | '); print(f'HOTA'.ljust(10), end=' | '); print(f'MOTA'.ljust(10), end=' | '); print(f'MOTP'.ljust(10), end=' | '); print(f'IDSW'.ljust(10), end=' | '); print(f'IDF1'.ljust(10), end=' | '); print(f'GT_IDs'.ljust(10), end=' | '); print(f'IDs'.ljust(10), end=' | '); print(f'TRCount'.ljust(10), end=' | '); print(f'TRRelCount'.ljust(10))

print('-' * 80)

import numpy as np

for alg, st in stats.items():
    
    print(f'{alg}'.ljust(10), end=' | ')

    for k, v in st.items():
        if k == 'GT_IDs' or k == 'IDs':
            print(f'{np.sum(np.array(v)):.2f}'.ljust(10), end=' | ')
        elif k == 'IDSW':
            print(f'{np.mean(np.array(v)):.2f}'.ljust(10), end=' | ')
        else:
            print(f'{np.mean(np.array(v)):.2f}'.ljust(10), end=' | ')

    print()


    
