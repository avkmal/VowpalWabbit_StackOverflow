from sklearn.metrics import accuracy_score
import numpy as np
from time import time
import os

def test_vw_model(model_filename, test_vw_file, prediction_filename,
                  true_labels, seed=17, quiet=True):

    init_time = time()
    vw_call_string = (f"vw -t -i {model_filename} {test_vw_file} " 
                       f"-p {prediction_filename} --random_seed {seed}")
    if quiet:
        vw_call_string += ' --quiet'
    
    print("\n")
    print(f"Testing...")    
    res = os.system(vw_call_string)
    
    if not res: 
        vw_pred = np.loadtxt(prediction_filename)
        accuracy = round(100 * accuracy_score(true_labels, vw_pred), 2)
        print(f"""
        model name: {model_filename}
        Accuracy: {accuracy}%. 
        Elapsed: {round(time() - init_time, 2)} sec.\n""")
        return accuracy
        print("---------------------------------------------------------")

    else:
        print('Failed.')
