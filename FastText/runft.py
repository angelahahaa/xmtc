import fasttext
TRAIN_DIR = '../data/amazon.train.txt'
lr = 0.1
epoch = 25
model = fasttext.train_supervised(
    input=TRAIN_DIR,
    epoch=epoch,
    lr=lr,
    wordNgrams=2,
    minCount=1,
    loss = 'ova',
    )
model.save_model("models/epoch{}_lr{}.bin".format(epoch,lr))
