# This script is used to train the batch model for all classifiers with different hyperparameters
# The training process is run in the background and the log is saved in the log file

# SVM
# C = [0.01, 0.1, 1, 10, 100]
echo "train svm with C=0.01"
nohup python train.py --classifier svm --C 0.01 > SVM_C_001.log &

echo "train svm with C=0.1"
nohup python train.py --classifier svm --C 0.1 > SVM_C_01.log &

echo "train svm with C=1"
nohup python train.py --classifier svm --C 1 > SVM_C_1.log &

echo "train svm with C=10"
nohup python train.py --classifier svm --C 10 > SVM_C_10.log &

echo "train svm with C=100"
nohup python train.py --classifier svm --C 100 > SVM_C_100.log &

# KNN
# k = [3, 5, 7, 9]
echo "train knn with k=3"
nohup python train.py --classifier knn --k 3 > KNN_k_3.log &

echo "train knn with k=5"
nohup python train.py --classifier knn --k 5 > KNN_k_5.log &

echo "train knn with k=7"
nohup python train.py --classifier knn --k 7 > KNN_k_7.log &

echo "train knn with k=9"
nohup python train.py --classifier knn --k 9 > KNN_k_9.log &

# Decision Tree
echo "train decision tree"
nohup python train.py --classifier decision_tree > Decision_Tree.log &