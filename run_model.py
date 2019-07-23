import numpy as np
import os,re,datetime
import pandas as pd
import pickle

from keras.preprocessing.text import Tokenizer
from keras.metrics import categorical_accuracy, binary_accuracy
from keras.callbacks import CSVLogger
import tensorflow as tf
import keras.backend as K

from sklearn.preprocessing import MultiLabelBinarizer

import scipy.sparse
from tools.helper import MetricsAtTopK, clean_str
from tools.MyClock import MyClock
from models import get_model
clk = MyClock()

# argparse
import argparse
parser = argparse.ArgumentParser(description = 'run baseline models')
parser.add_argument('-i','--input', required = True, type = str, help = 'input directory e.g. ./data/dl_amazon_1/')
parser.add_argument('-o','--output', required = True, type = str, help = 'output directory')
parser.add_argument('-m','--model', required = True, type = str, help = 'model, one in: xmlcnn, attentionxml, attention,')
parser.add_argument('--epoch', default = 5, type = int, help = 'epochs')
parser.add_argument('--batch_size', default = 0, type = int, help = 'batch size')
parser.add_argument('--early_stopping', default = False, action = 'store_true', help = 'early stopping using validation set (not implemented yet)')
parser.add_argument('--save_weights', default = True, action = 'store_true', help = 'save trained model weights')
parser.add_argument('--save_prediction', default = 10, type = int, help = 'save top k prediction and corresponding probabilities (not implemented yet)')
args = parser.parse_args()

def binary_cross_entropy_with_logits(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true,y_pred,from_logits=True),axis=-1)
# metrics
def binary_accuracy_with_logits(y_true, y_pred):
    return K.mean(K.equal(y_true, K.tf.cast(K.less(0.0,y_pred), y_true.dtype)))
pat1 = MetricsAtTopK(k=1)
pat5 = MetricsAtTopK(k=5)
def p1(x,y):
    return pat1.precision_at_k(x,y)
def p5(x,y):
    return pat5.precision_at_k(x,y)

if not args.batch_size:
    if args.model == 'attention':
        args.batch_size = 25
    elif args.model == 'xmlcnn':
        args.batch_size = 128
    elif args.model == 'attentionxml':
        args.batch_size = 20

IN_DIR = args.input
OUT_DIR = args.output
in_dirs = {
    'embedding_matrix':'embedding_matrix.npy',
    'x_train':'x_train.npy',
    'x_test':'x_test.npy',
    'y_train':'y_train.npz',
    'y_test':'y_test.npz'
}
for key,val in in_dirs.items():
    d = os.path.join(IN_DIR,val)
    if not os.path.exists(d):
        raise Exception('path does not exist: {}'.format(d))
    else:
        in_dirs[key] = d
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
out_dir = os.path.join(
    args.output,
    datetime.datetime.now().strftime('%y%m%d_%H%M%S_{}'.format(args.model)),
)

# things
if not os.path.exists(IN_DIR):
    raise Exception('input path does not exist: {}'.format(IN_DIR))
print('READ DATA...')
embedding_matrix = np.load(in_dirs['embedding_matrix'])
x_train = np.load(in_dirs['x_train'])
x_test = np.load(in_dirs['x_test'])
y_train = scipy.sparse.load_npz(in_dirs['y_train'])
y_test = scipy.sparse.load_npz(in_dirs['y_test'])
y_train = y_train.todense()
y_test = y_test.todense()
labels_dim = y_train.shape[-1]
num_words,embedding_dim = embedding_matrix.shape
max_sequence_length = x_train.shape[1]
print('Train: {}, Test: {}, Labels: {}, Vocab size: {}, Embedding: {}'.format(
    x_train.shape[0],x_test.shape[0],labels_dim,num_words-1,embedding_dim))

model = get_model(args.model,embedding_matrix,max_sequence_length,labels_dim)
model.compile(loss=binary_cross_entropy_with_logits,
              optimizer='adam',
              metrics=[binary_accuracy_with_logits,p1,p5])

print(model.summary())
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
csv_logger = CSVLogger(os.path.join(out_dir,'train.log'),append=False)
if args.early_stopping:
    pass
else:
    model.fit(x_train, y_train,
              batch_size = args.batch_size,
              epochs = args.epoch,
              validation_data = (x_test, y_test),
              callbacks = [csv_logger],
              shuffle=True,
             )
if args.save_weights:
    model.save_weights(os.path.join(out_dir,'weights.h5'))
if args.save_prediction:
    print('SAVE PREDICTIONS')
    k = args.save_prediction
    batch_size = x_test.shape[0]//100
    IND_DIR = os.path.join(out_dir,'prediction_{}_ind.txt'.format(k))
    LOGITS_DIR = os.path.join(out_dir,'prediction_{}_logits.txt'.format(k))
    f_ind = open(IND_DIR,'ab')
    f_logits = open(LOGITS_DIR,'ab')
    s = x_test.shape[0]
    clk.tic()
    for i,start in enumerate(range(0,s,batch_size)):
        end = min(start+batch_size,s)
        x_batch = x_test[start:end,:]
        out_probs = model.predict(x_batch)
        ind = np.argsort(out_probs,axis=1)[:,-k:]
        ind = ind[:,::-1]
        logits = np.take_along_axis(out_probs, ind, axis=1)
        np.savetxt(f_ind,ind,fmt='%d')
        np.savetxt(f_logits,logits,fmt='%1.3f')
        print('{:0.0f}% {}'.format(end/s*100,clk.toc(False)),end='\r')
    f_ind.close()
    f_logits.close()
csv_path = os.path.join(out_dir,'args.csv')
pd.DataFrame.from_dict([vars(args)]).to_csv(csv_path)
