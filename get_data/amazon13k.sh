#!/bin/bash
fileid="0B3lPMIHmG6vGYm9abnAzTU1XaTQ"
filename="Amazon_RawData.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
