import os
from tqdm import tqdm
from time import time
import numpy as np
import sys
from tqdm import tqdm

topics = ['javascript', 'java', 'python', 'ruby', 'php',
          'c++', 'c#', 'go', 'scala', 'swift']
topic_set = set(topics)
topic_map = dict(zip(topics, range(1, len(topics) + 1)))

num_corrupted, num_selected = 0, 0
with open('data/' + sys.argv[1]) as inp_file, open('data/' + sys.argv[2], 'w') as out_file:
    for line in tqdm(inp_file):
        values = line.strip().split('\t')
        if len(values) != 2:
            num_corrupted += 1
            continue
        text, labels = values
        labels = set(labels.split())
        topics_from_list = labels.intersection(topic_set)
        if len(topics_from_list) == 1:
            num_selected += 1
            out_file.write('{} | {}\n'.format(str(topic_map[list(topics_from_list)[0]]), 
                                              text.strip().replace(':', '').replace('|', '')))
print(f"{num_selected} lines selected, {num_corrupted} lines corrupted.")


