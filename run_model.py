#!/usr/bin/env python
# coding: utf-8

# # IMPORTS




# basic
import argparse
import os,datetime

# save things
import pandas as pd
from keras.callbacks import CSVLogger

# model_func
from tools.model_func import *








# # ARGPARSE




parser = argparse.ArgumentParser(description = 'run baseline models')
parser.add_argument('-i','--input', required = True, type = str, help = 'input directory e.g. ./data/dl_amazon_1/')
parser.add_argument('-m','--model', required = True, type = str, help = 'model, one in: xmlcnn, attentionxml, attention,')
parser.add_argument('-l','--loss', required = True, type = str, help = "loss type, categorical or binary ")
parser.add_argument('-o','--output', required = True, type = str, help = 'output directory')
parser.add_argument('--mode', required = True, type = str, help = 'cat,hierarchy')
parser.add_argument('--epoch', default = 5, type = int, help = 'epochs')
parser.add_argument('--batch_size', default = 0, type = int, help = 'batch size')
parser.add_argument('--save_weights', default = True, action = 'store_true', help = 'save trained weights')
parser.add_argument('--save_model', default = True, action = 'store_true', help = 'save trained model architecture')
parser.add_argument('--save_prediction', default = True, action = 'store_true', help = 'save top 10 prediction and corresponding probabilities (not implemented yet)')
args = parser.parse_args()

# argparse validation
default_batch_size = {'xmlcnn':128,'attentionxml':20,'attention':25}
if not os.path.exists(args.input):
    raise Exception('Input path does not exist: {}'.format(args.input))
if args.model not in default_batch_size.keys():
    raise Exception('Unknown model: {}'.format(args.model))
if args.loss not in ['binary','categorical']:
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





# # MAIN




csv_logger = CSVLogger(os.path.join(OUT_DIR,'train.log'),append=False)
# inputs
x_train,y_trains,x_test,y_tests = get_input(IN_DIR,args.mode)
_,max_sequence_length = x_train.shape
labels_dims = [l.shape[-1] for l in y_tests]
# model
embedding_layer = get_embedding_layer(IN_DIR)
model = get_model(model_name = args.model,
                  max_sequence_length = max_sequence_length,
                  labels_dims = labels_dims,
                  embedding_layer = embedding_layer)
# train 
model.compile(loss = loss_dict[args.loss],
              optimizer = 'adam',
              metrics = [pAt1,pAt5])
model.fit(x_train, y_trains,
          batch_size = args.batch_size,
          epochs = args.epoch,
          validation_data = (x_test, y_tests),
          callbacks = [csv_logger],
          shuffle = True,
         )


# # save things




if args.save_weights:
    model.save_weights(os.path.join(OUT_DIR,'weights.h5'))
if args.save_model:
    with open(os.path.join(OUT_DIR,'model.json'),'w') as f:
        f.write(model.to_json())
if args.save_prediction:
    save_predictions(model,x_test,y_tests,OUT_DIR)
pd.DataFrame.from_dict([vars(args)]).to_csv(os.path.join(OUT_DIR,'args.csv'))
