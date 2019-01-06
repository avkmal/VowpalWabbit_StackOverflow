import sys
sys.path.insert(0, '/home/akmal/Desktop/VowpalWabbit_StackOverflow/Training')
from training import train_vw_model

import sys
sys.path.insert(0, '/home/akmal/Desktop/VowpalWabbit_StackOverflow/Validation')
from testing import test_vw_model    
import numpy as np

y_test = np.loadtxt('data/stackoverflow_test_labels.txt')


train_vw_model('data/stackoverflow_train_valid.vw', 
                'data/vw_model10.vw', 
                ngram=2, passes=1,
                num_classes=10, bit_precision=28, 
                seed=17, quiet=True)

test_vw_model(model_filename = 'data/vw_model10.vw', 
            test_vw_file = 'data/stackoverflow_test.vw', 
            prediction_filename = 'data/vw_test_pred10.csv',
            true_labels=y_test, seed=17)