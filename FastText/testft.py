import fasttext
import re
import argparse
parser = argparse.ArgumentParser(description='run FastText')
parser.add_argument('-m','--model',help='model directory e.g. models/epoch5_lr0.1.bin')
parser.add_argument('-d','--data',help='data directory e.g. .../data/amazon.test.txt')
parser.add_argument('-k',default=1,type=int)
parser.add_argument('--one_class',action='store_true', default=False)
args = parser.parse_args()

MODEL_DIR = args.model
TEST_DIR = args.data
k = args.k
# start
model = fasttext.load_model(MODEL_DIR)
def get_pAtk(correct,samples,k,one_class = False):
    if one_class:
        return correct/((i+1))*100
    else:
        return correct/((i+1)*args.k)*100

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
correct = 0
samples = len(contents)
disp = samples//10
print('Calculate p@{}'.format(k))
for i in range(samples):
    pred = model.predict(contents[i],k=args.k)
    for p in pred[0]:
        if p in labels[i]:
            correct+=1
    if i%disp==0:
        c = (i+1)/(samples)*100
        p = get_pAtk(correct,samples,args.k,args.one_class)
        print('TEST {:.2f}%, P@1 = {:.2f}'.format(c,p))
p = get_pAtk(correct,samples,args.k,args.one_class)
print('COMPLETED, P@k = {:.2f}'.format(p))
