import numpy as np
import os,re,datetime
import pandas as pd
import pickle
from models import get_model
from tools.MyClock import MyClock
import scipy.sparse
import warnings
warnings.filterwarnings('ignore')
clk = MyClock()


k=10
# start
IN_DIR = 'data/dl_amazon_1h'
MODEL_DIRS = [
        # 'outputs/190722_204831_xmlcnn',
        # 'outputs/190715_215817_attentionxml',
        # 'outputs/190715_235041_attention',
        'outputs/190716_105336_xmlcnn',
        'outputs/190716_112050_attentionxml',
        'outputs/190718_125651_attention',
        ]

print('READ DATA...')
in_dirs = {
    'embedding_matrix':'embedding_matrix.npy',
    'x_test':'x_test.npy',
    'y_test':'y_test.npz'
}
for key,val in in_dirs.items():
    d = os.path.join(IN_DIR,val)
    if not os.path.exists(d):
        raise Exception('path does not exist: {}'.format(d))
    else:
        in_dirs[key] = d
embedding_matrix = np.load(in_dirs['embedding_matrix'])
x_test = np.load(in_dirs['x_test'])
y_test = scipy.sparse.load_npz(in_dirs['y_test'])
y_test = y_test.todense()
labels_dim = y_test.shape[-1]
num_words,embedding_dim = embedding_matrix.shape
max_sequence_length = x_test.shape[1]


for MODEL_DIR in MODEL_DIRS:
    MODEL_NAME = MODEL_DIR.split('_')[-1]
    WEIGHT_DIR = os.path.join(MODEL_DIR,'weights.h5')
    print('Running on : {}'.format(MODEL_DIR))

    model = get_model(MODEL_NAME,embedding_matrix,max_sequence_length,labels_dim)
    model.load_weights(WEIGHT_DIR)
    batch_size = x_test.shape[0]//100
    inds = []
    probs = []
    s = x_test.shape[0]
    clk.tic()
    for i,start in enumerate(range(0,s,batch_size)):
        end = min(start+batch_size,s)
        x_batch = x_test[start:end,:]
        out_probs = model.predict(x_batch)
        ind = np.argsort(out_probs,axis=1)[:,-k:]
        ind = ind[:,::-1]
        probs.append(np.take_along_axis(out_probs, ind, axis=1))
        inds.append(ind)
        print('{:0.0f}% {}'.format(end/s*100,clk.toc(False)),end='\r')

    DIR = os.path.join(MODEL_DIR,'prediction_{}_ind.txt'.format(k))
    with open(DIR,'ab') as f:
        for ind in inds:
            np.savetxt(f,ind,fmt='%d')
    DIR = os.path.join(MODEL_DIR,'prediction_{}_logits.txt'.format(k))
    with open(DIR,'ab') as f:
        for prob in probs:
            np.savetxt(f,prob,fmt='%1.3f')
