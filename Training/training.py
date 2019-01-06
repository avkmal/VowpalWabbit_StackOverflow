import numpy as np
from time import time
import os

def train_vw_model(train_vw_file, model_filename, num_classes=10,
                   ngram=1, loss_function='hinge',
                   bit_precision=28, passes=1,
                   seed=17, quiet=True):
                   
    init_time = time()
    vw_call_string = (f'vw --oaa {num_classes} {train_vw_file} '  
                      f'-f {model_filename} -b {bit_precision} ' 
                      f'--loss_function {loss_function} --random_seed {seed}')

    if ngram > 1:
         vw_call_string += f" --ngram={ngram}"
            
    if passes > 1:
         vw_call_string += \
         f" -k --passes={passes} --cache_file {model_filename.replace('.vw', '.cache')}"
                            
    if quiet:
        vw_call_string += ' --quiet'

    
    print('\n')
    print(f"Training (ngrams:{ngram}, passes:{passes})...")
   
    res = os.system(vw_call_string)

    print(f"""
    Success. Elapsed: {round(time() - init_time, 2)} sec"""
             if not res else 'Failed.')

    return ngram, passes
    

