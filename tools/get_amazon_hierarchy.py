import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict,Counter
import matplotlib.pyplot as plt
from tools.MyClock import MyClock
from tools.helper import clean_str
clk = MyClock()
# import
DATA_DIR = 'data/amazon_des.pkl'
df = pd.read_pickle(DATA_DIR)

# read categories (hierarchy)
CAT_DIR = './Amazon_RawData/categories.txt'
CAT_MAP_DIR = "./AmazonCat-13K_mappings/AmazonCat-13K_label_map.txt"
with open(CAT_MAP_DIR,'r',encoding = "ISO-8859-1") as f:
    cats_map = f.read().splitlines()
cats_set = set([c.lower() for c in cats_map])
with open(CAT_DIR,'r',encoding = "ISO-8859-1") as f:
    lines = f.read().splitlines()
data = defaultdict(list)
invalid = []
for i,line in enumerate(lines):
    if line[0]!=' ':
        id = line
    elif line[:2]=='  ':
        cats = [t.strip().lower() for t in line.split(',')]
        cats = [c for c in cats if c in cats_set]
        if cats:
            data[id].append(cats)
        else:
            invalid.append((i,line))
    else:
        raise Exception('invalide line {} : {}'.format(i,line))
data = {key:data[key] for key in df.index.values}

# randomly chose one hierachy
optimal_depth = 0
one_h_data = defaultdict(list)
for id, texts in data.items():
    if len(texts) == 1:
        ind = 0
    else:
        length = min([max([len(text) for text in texts]),optimal_depth])
        texts = [text for text in texts if len(text)>=length]
        ind = np.random.randint(len(texts))
    one_h_data[id] = texts[ind]

# clip extreme depths
q = int(np.percentile([len(x) for x in one_h_data.values()],95))
for key,val in one_h_data.items():
    i = int(min(len(val),q))
    one_h_data[key] = ' > '.join(val[:i])
print('clip depth to {}'.format(q))
df['categories'] = df.index.map(one_h_data)

# clean text
clk.tic()
cleaned = {}
aa = len(df)
for i,(id,text) in enumerate(df['text'].iteritems()):
    if id in cleaned.keys():
        continue
    cleaned[id] = clean_str(text)
    if i%(aa//100)==0:
        print(clk.toc(False))
        print('COMPLETE {:.0f}% '.format(i/aa*100),end = '')
    elif i%(aa//1000)==0:
        print('.',end = '')

df['text'] = df.index.map(cleaned)

print(df['train/test'].value_counts())

df.to_pickle('data/amazon_1h.pkl')
