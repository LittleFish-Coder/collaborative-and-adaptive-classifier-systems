# !/bin/bash
# if dataset.zip is not present, download it
if [ ! -f dataset.zip ]; then
    echo "Downloading dataset.zip"
    curl -L -o dataset.zip https://www.kaggle.com/api/v1/datasets/download/arjunashok33/miniimagenet
else
    echo "dataset.zip already exists"
fi

# unzip the dataset
echo "Unzipping dataset.zip to miniimagenet folder"
unzip dataset.zip -d miniimagenet