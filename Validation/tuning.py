import sys
sys.path.insert(0, '/home/akmal/Desktop/VowpalWabbit_StackOverflow/Training')

from training import train_vw_model
from testing import test_vw_model
import itertools
from tqdm import tqdm
import numpy as np

y_valid = np.loadtxt('data/stackoverflow_valid_labels.txt')
y_test = np.loadtxt('data/stackoverflow_test_labels.txt')


results = []
for i, (ngram, passes) in tqdm(enumerate([(i,x) for i in [1,2,3] for x in [1,3,5]])):
    
    ngram, passes = train_vw_model('data/stackoverflow_train.vw',
                                    f"data/vw_model{i+1}.vw", 
                                    ngram=ngram, passes=passes,
                                    loss_function='hinge',
                                    num_classes=10, bit_precision=28, 
                                    seed=17, quiet=True)

    accuracy = test_vw_model(model_filename = f"data/vw_model{i+1}.vw", 
                            test_vw_file = 'data/stackoverflow_valid.vw', 
                            prediction_filename = f"data/vw_valid_pred{i+1}.csv",
                            true_labels=y_valid, seed=17)

    results.append((ngram, passes, accuracy))

results.sort(key = lambda x:x[-1], reverse=True)

best_ngram, best_passes, best_accuracy = results[0] 

print(f"""
Complete validation.
Highest accuracy: {best_accuracy}%
ngrams:{best_ngram}, passes: {best_passes}
""")
