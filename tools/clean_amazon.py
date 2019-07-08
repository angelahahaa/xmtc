# manually checked that all train test ids in the feature version is a subset of the raw version
# ids are unique in all file
# feature version categories are a subset of the cleaned raw category

import joblib,re,os
import numpy as np
import pandas as pd
import collections
import warnings
warnings.filterwarnings("ignore")

# file directories
NAME_DIR = './Amazon_RawData/titles.txt'
DES_DIR = './Amazon_RawData/descriptions.txt'
CAT_DIR = './Amazon_RawData/categories.txt'
TRAIN_DIR = "./AmazonCat-13K_mappings/AmazonCat-13K_train_map.txt"
TEST_DIR = "./AmazonCat-13K_mappings/AmazonCat-13K_test_map.txt"
CAT_MAP_DIR = "./AmazonCat-13K_mappings/AmazonCat-13K_label_map.txt"
OUT_DIR = "./data/amazon_des.pkl"

# define id pattern
pattern = re.compile("[^A-Z0-9]+")

# create dictionary of id
data = collections.defaultdict(dict)

print('READING DATA...')
# read names
with open(NAME_DIR,'r',encoding = "ISO-8859-1") as f:
    lines = f.read().splitlines()
for i,line in enumerate(lines):
    newid = True
    if len(line)<11:
        newid = False
    elif line[10]!=' ' or pattern.search(line[:10]):
        newid = False
    else:
        id = line[:10]
    if newid:
        data[id]['name']=line[11:]
    else:
        data[id]['name'] = data[id]['name']+' '+line

# read descriptions
with open(DES_DIR,'r',encoding = "ISO-8859-1") as f:
    lines = f.read().splitlines()
for i,line in enumerate(lines):
    if not line:
        continue
    if line[:18]=='product/productId:':
        _,id = line.split(' ')
    elif line[:20] == 'product/description:':
        _,description = line.split(' ',1)
        data[id]['description'] = description
    else:
        data[id]['description'] = data[id]['description']+" "+line

# read categories
with open(CAT_MAP_DIR,'r',encoding = "ISO-8859-1") as f:
    cats_map = f.read().splitlines()
cats_set = set([c.lower() for c in cats_map])
with open(CAT_DIR,'r',encoding = "ISO-8859-1") as f:
    lines = f.read().splitlines()
for i,line in enumerate(lines):
    if line[0]!=' ':
        id = line
        data[id]['categories']=[]
    elif line[:2]=='  ':
        cats = [t.strip().lower() for t in line.split(',')]
        cats = [c for c in cats if c in cats_set]
        data[id]['categories'] = data[id]['categories']+ cats
    else:
        raise Exception('invalide line {} : {}'.format(i,line))

print('OBTAIN TEST TRAIN ID..')
# read train test id
traindf = pd.read_csv(
    TRAIN_DIR,
    sep=r'->',
    header=None,
    names=['id','title_mappings'],
)
traindf = traindf.drop(columns = 'title_mappings')
testdf = pd.read_csv(
    TEST_DIR,
    sep=r'->',
    header=None,
    names=['id','title_mappings'],
)
testdf = testdf.drop(columns = 'title_mappings')

trainid = set(traindf.id.to_list())
testid = set(testdf.id.to_list())
d = {}
for id in trainid:
    data[id]['train/test'] = 'train'
    d[id] = data[id]
for id in testid:
    data[id]['train/test']='test'
    d[id] = data[id]

# create dataframe
print('CREATE DATAFRAME..')
df = pd.DataFrame.from_dict(d, orient='index')

# # combine name and description with seperator |||
# df['name'] = df['name'].fillna('none')
# df['text'] = df['name'] + " ||| " + df['description']
# df = df.drop(columns = ['name','description'])
# print(df['train/test'].value_counts())
df['text'] = df['description']
df = df.drop(columns = ['name','description'])
print(df['train/test'].value_counts())
# save as pkl
print('SAVE DATAFRAME..')
df.to_pickle(OUT_DIR)
