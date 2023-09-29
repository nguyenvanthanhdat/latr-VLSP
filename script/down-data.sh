mkdir data

cd data
wget -nc https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json
wget -nc https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json
wget -nc https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_train.json
wget -nc https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_val.json
wget -nc https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip

# unzip data file
# tar -xf train_val_images.zip
python -m zipfile -e train_val_images.zip .