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

- Train the model (SVM, KNN, Decision Tree)
    
```bash
python train.py
```

```bash
python train.py --classifier <classifier> --C <C> --k <k>
# choose classifier from `svm`, `knn`, `decision_tree`
# choose C from [0.01, 0.1, 1, 10, 100]
# choose k from [1, 3, 5, 7, 9]
```

This shell script will train the model with all possible combinations of C and k.

**Note: The process will run in the background (kill the process if needed)**
```bash
bash train_batch_model.sh
```

- Test the model (from weight)
```bash
python test.py --model_path <model_path>
```