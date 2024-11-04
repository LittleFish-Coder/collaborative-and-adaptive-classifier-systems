import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import os
# scikit-learn SVM, KNN, DecisionTree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from argparse import ArgumentParser

class ImageDataset(Dataset):
    def __init__(self, dataset_dir='dataset', annotaion='train'):
        self.image_paths = []
        self.images = []
        self.labels = []
        self.transform = Compose([Resize((32, 32)), ToTensor()])

        # read from the annotation file from the dataset directory
        with open(f'{dataset_dir}/{annotaion}.csv', 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # skip the header
                image_path, label = line.strip().split(',')
                self.image_paths.append(os.path.join(dataset_dir, image_path))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        image = Image.open(self.image_paths[idx])
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

def fit_svm(train_dataset, C=1):
    # Initialize the SVM classifier
    clf = SVC(kernel='linear', C=C)
    # Train the SVM classifier
    X = []
    y = []
    for image, label in train_dataset:
        X.append(image.view(-1).numpy())
        y.append(label)
    clf.fit(X, y)
    return clf

def fit_knn(train_dataset, k=3):
    # Initialize the KNN classifier
    clf = KNeighborsClassifier(n_neighbors=k)
    # Train the KNN classifier
    X = []
    y = []
    for image, label in train_dataset:
        X.append(image.view(-1).numpy())
        y.append(label)
    clf.fit(X, y)
    return clf

def fit_decision_tree(train_dataset):
    # Initialize the Decision Tree classifier
    clf = DecisionTreeClassifier()
    # Train the Decision Tree classifier
    X = []
    y = []
    for image, label in train_dataset:
        X.append(image.view(-1).numpy())
        y.append(label)
    clf.fit(X, y)
    return clf

def test_classifier(clf, test_dataset):
    correct = 0
    total = 0
    for image, label in test_dataset:
        X = image.view(-1).numpy()
        predicted = clf.predict([X])[0]
        if predicted == label:
            correct += 1
        total += 1
    accuracy = correct / total
    return accuracy

if __name__ == '__main__':
    train_dataset = ImageDataset(annotaion='train')
    print(f'Train dataset size: {len(train_dataset)}')

    # Train the SVM classifier
    svm = fit_svm(train_dataset)

    # test the SVM classifier
    test_dataset = ImageDataset(annotaion='test')
    print(f'Test dataset size: {len(test_dataset)}')
    accuracy = test_classifier(svm, test_dataset)
    print(f'SVM accuracy: {accuracy}')