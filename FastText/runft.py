import fasttext
import argparse

parser = argparse.ArgumentParser(description='run FastText')
parser.add_argument('-i','--input',required = True,help='input directory e.g. ./data/amazon.train.txt')
parser.add_argument('-o','--output',required = True,help='output directory for model e.g. ./FastText/models/amazon_')
parser.add_argument('--lr',type=float,default=0.1,help='learning rate')
parser.add_argument('--epoch',type=int,default=5,help='epochs')
args = parser.parse_args()

TRAIN_DIR = args.input
lr = args.lr
epoch = args.epoch
model = fasttext.train_supervised(
    input=TRAIN_DIR,
    epoch=epoch,
    lr=lr,
    wordNgrams=2,
    minCount=1,
    loss = 'ova',
    )
save_path = "{}_epoch{}_lr{}.bin".format(args.output,epoch,lr)
model.save_model(save_path)
print('Model saved to:\n{}'.format(save_path))
