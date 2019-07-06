import fasttext
import re
import argparse
parser = argparse.ArgumentParser(description='run FastText')
parser.add_argument('-m','--model',help='model directory e.g. models/epoch5_lr0.1.bin')
parser.add_argument('-d','--data',help='data directory e.g. .../data/amazon.test.txt')
args = parser.parse_args()

MODEL_DIR = args.model
TEST_DIR = args.data
# start
model = fasttext.load_model(MODEL_DIR)

with open(TEST_DIR,'r',encoding = "ISO-8859-1") as f:
    lines = f.read().splitlines()
# get labels and contents
label_pattern = re.compile('__label__\S+')
labels = []
contents = []
for line in lines:
    labs = set(label_pattern.findall(line))
    labels.append(labs)
    contents.append(label_pattern.sub(r'',line).strip())

# predict and get P@k
correct = 0
samples = len(contents)
disp = samples//10
print('Runnint test')
for i in range(samples):
    pred = model.predict(contents[i])
    if pred[0][0] in labels[i]:
        correct += 1
    if i%disp==0:
        c = (i+1)/samples*100
        p = correct/(i+1)*100
        print('TEST {:.2f}%, P@1 = {:.2f}'.format(c,p))
p = correct/samples*100
print('COMPLETED, P@k = {:.2f}'.format(p))