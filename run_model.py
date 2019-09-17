#!/usr/bin/env python
# coding: utf-8

# # IMPORTS




# basic
import argparse
import os,datetime

# save things
import pandas as pd
from tensorflow.keras.callbacks import CSVLogger

# model_func
from tools.model_func import *








# # ARGPARSE




parser = argparse.ArgumentParser(description = 'run baseline models')
parser.add_argument('-i','--input', required = True, type = str, help = 'input directory e.g. ./data/dl_amazon_1/')
parser.add_argument('-m','--model', required = True, type = str, help = 'model, one in: xmlcnn, attentionxml, attention,')
parser.add_argument('-l','--loss', required = True, type = str, help = "loss type: categorical, binary, masked_categorical ")
parser.add_argument('-o','--output', required = True, type = str, help = 'output directory')
parser.add_argument('--mode', required = True, type = str, help = 'cat,hierarchy')
parser.add_argument('--epoch', default = 5, type = int, help = 'epochs')
parser.add_argument('--batch_size', default = 0, type = int, help = 'batch size')
parser.add_argument('--save_weights', default = True, action = 'store_true', help = 'save trained weights')
parser.add_argument('--save_model', default = False, action = 'store_true', help = 'save trained model architecture')
parser.add_argument('--save_prediction', default = 1, type = int, help = 'save top 10 prediction and corresponding probabilities')
parser.add_argument('--gpu', default = '', type = str, help = 'GPU id to use')
parser.add_argument('--bert_bottle_neck', default = 512, type = int, help = 'bottle neck dim for bert, 0 implies no bottle neck layer')
parser.add_argument('--bert_trainable_layers', default = 10, type = int, help = 'number of trainable layers in bert ')
parser.add_argument('--val', default = False, action = 'store_true',help = 'use validation set')
args = parser.parse_args()

# argparse validation
default_batch_size = {'xmlcnn':128,'attentionxml':20,'attention':25,'bert':256,}
if not os.path.exists(args.input):
    raise Exception('Input path does not exist: {}'.format(args.input))
if args.model not in default_batch_size.keys():
    raise Exception('Unknown model: {}'.format(args.model))
if args.loss not in ['binary','categorical','masked_categorical']:
    raise Exception('Unknown loss: {}'.format(args.loss))
if args.mode not in ['cat','hierarchy']:
    raise Exception('Unknown mode: {}'.format(args.mode))

IN_DIR = args.input
if not args.batch_size:
    args.batch_size = default_batch_size[args.model]
if not os.path.exists(args.output):
    os.mkdir(args.output)
    print(Coloured("Create Output Directory: {}".format(args.output)))
OUT_DIR = os.path.join(
    args.output,
    datetime.datetime.now().strftime('%y%m%d_%H%M%S_{}'.format(args.model)),
)
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

if args.gpu:
    print(Coloured("USE GPU: {}".format(args.gpu)))
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu





# # MAIN


pd.DataFrame.from_dict([vars(args)]).to_csv(os.path.join(OUT_DIR,'args.csv'))

# inputs
if args.model == 'bert':
    x_trains,y_trains,x_tests,y_tests = get_bert_input(IN_DIR,args.mode)
else:
    x_trains,y_trains,x_tests,y_tests = get_input(IN_DIR,args.mode)
if args.loss.startswith('masked'):
    print(Coloured("MASKING INPUT"))
    y_trains = mask_ys(y_trains,IN_DIR)
    y_tests = mask_ys(y_tests,IN_DIR)
max_sequence_length = len(x_trains[0][0])
labels_dims = [l.shape[-1] for l in y_tests]

if args.val:
    print(Coloured("Use Validation"))
    x_trains,y_trains,x_vals,y_vals = get_unbiased_train_val_split(x_trains,y_trains,IN_DIR)
else:
    x_vals,y_vals = x_tests,y_tests
del x_tests
del y_tests
# loss
loss_dict = {'binary':binary_cross_entropy_with_logits,
             'categorical':categorical_cross_entropy_with_logits,
             'masked_categorical':masked_categorical_cross_entropy_with_logits,
            }

# callbacks
callbacks = []
csv_log_dir = os.path.join(OUT_DIR,'train.log')
callbacks.append(CSVLogger(csv_log_dir,append=False))

# optimizers
if args.model == 'bert':
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
else:
    optimizer = 'adam'

# build model
sess = tf.Session()
if args.model == 'bert':
    model = get_bert_model(max_sequence_length, labels_dims,
                        bottle_neck = args.bert_bottle_neck,
                        trainable_layers = args.bert_trainable_layers,
                        sess = sess,
                        )
else:
    embedding_layer = get_embedding_layer(IN_DIR)
    model = get_model(model_name = args.model,
                      max_sequence_length = max_sequence_length,
                      labels_dims = labels_dims,
                      embedding_layer = embedding_layer)
# print summary
model.summary()

# compile
model.compile(loss = loss_dict[args.loss],
              optimizer = optimizer,
              metrics = [pAt1,pAt5])
# train
print(Coloured("TRAIN"))
model.fit(x_trains, y_trains,
          batch_size = args.batch_size,
          epochs = args.epoch,
          validation_data = (x_vals, y_vals),
          callbacks = callbacks,
          shuffle = True,
         )
# # save things

if args.save_weights:
    model.save_weights(os.path.join(OUT_DIR,'weights.h5'))
if args.save_model:
    with open(os.path.join(OUT_DIR,'model.json'),'w') as f:
        f.write(model.to_json())
if args.save_prediction:
    if args.model == 'bert':
        _,_,x_tests,y_tests = get_bert_input(IN_DIR,args.mode)
    else:
        _,_,x_tests,y_tests = get_input(IN_DIR,args.mode,get_output = [0,0,1,1])
    if args.loss == 'masked_categorical':
        save_hs_predictions(model,x_tests,y_tests,OUT_DIR,IN_DIR)
    else:
        save_predictions(model,x_tests,y_tests,OUT_DIR)
# evaluate
if args.val:
    print(Coloured("EVALUATE"))
    print('Get test data')
    if args.model == 'bert':
        _,_,x_tests,y_tests = get_bert_input(IN_DIR,args.mode)
    else:
        _,_,x_tests,y_tests = get_input(IN_DIR,args.mode,get_output = [0,0,1,1])
    print('Run test')
    test_results = model.evaluate(x_tests,y_tests)
    dd = {k:v for k,v in zip(model.metrics_names,test_results)}
    dd['epoch']= 'evaluate'
    df = pd.read_csv(csv_log_dir)
    df = df.append(dd,ignore_index=True)
    df.to_csv(csv_log_dir,index = False)
# close Session
sess.close()
