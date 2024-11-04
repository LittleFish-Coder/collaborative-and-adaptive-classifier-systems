# Collaborative and Adaptive Classifier Systems

NCKU 113 Fall - Machine Learning Assignment 1

## Usage

```bash
git clone https://github.com/LittleFish-Coder/collaborative-and-adaptive-classifier-systems.git
cd collaborative-and-adaptive-classifier-systems
```

- Install dependencies

```bash
pip install -r requirements.txt
```

- Download dataset (execute the command in `dataset` folder)

```bash
cd dataset
bash download_dataset.sh
```

- Generate train and test csv files

```bash
python generate_train_test.py
```

- Train the model (SVM, KNN, Decision Tree, Random Forest, AdaBoost, Gradient Boosting)
    
```bash
python train.py
```
