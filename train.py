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
import pickle

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

def show_args(args):
    print(f'---------- Arguments -----------')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print(f'-------------------------------')

def fit_classifier(train_dataset, classifier, C, k):
    if classifier == 'svm':
        return fit_svm(train_dataset, C=C)
    elif classifier == 'knn':
        return fit_knn(train_dataset, k=k)
    elif classifier == 'decision_tree':
        return fit_decision_tree(train_dataset)

def fit_svm(train_dataset, C=1):
    # Initialize the SVM classifier
    clf = SVC(kernel='rbf', C=C)
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
    # Collect true labels and predicted labels
    true_labels = []
    predicted_labels = []

    for image, label in test_dataset:
        X = image.view(-1).numpy()
        predicted = clf.predict([X])[0]
        true_labels.append(label)
        predicted_labels.append(predicted)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    return accuracy, precision, recall, f1

def save_model(model, output_dir, classifier, C, k):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # use pickle to save model
    if classifier == 'svm':
        output_path = f'{output_dir}/{classifier}_C_{C}.pkl'
    elif classifier == 'knn':
        output_path = f'{output_dir}/{classifier}_k_{k}.pkl'
    elif classifier == 'decision_tree':
        output_path = f'{output_dir}/{classifier}.pkl'
    pickle.dump(model, open(output_path, 'wb'))
    print(f'Model saved to {output_path}')

if __name__ == '__main__':

    # args
    parser = ArgumentParser()
    parser.add_argument('--classifier', type=str, default='svm', help='classifier to use', choices=['svm', 'knn', 'decision_tree'])
    parser.add_argument('--C', type=float, default=1, help='SVM regularization parameter', choices=[0.01, 0.1, 1, 10, 100])
    parser.add_argument('--k', type=int, default=3, help='KNN number of neighbors', choices=[3, 5, 7, 9])
    parser.add_argument('--output_dir', type=str, default='weights', help='directory to save the model')
    args = parser.parse_args()

    # show arguments
    show_args(args)
    classifier = args.classifier
    C = args.C
    k = args.k
    output_dir = args.output_dir

    # Load the dataset
    train_dataset = ImageDataset(annotaion='train')
    print(f'Train dataset size: {len(train_dataset)}')

    # Fit the classifier
    print(f'Fitting the classifier: {classifier}')
    model = fit_classifier(train_dataset, classifier, C, k)

    # Test the classifier
    test_dataset = ImageDataset(annotaion='test')
    print(f'Test dataset size: {len(test_dataset)}')
    accuracy, precision, recall, f1 = test_classifier(model, test_dataset)
    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')

    # save the model
    print(f'Saving the model to {output_dir}')
    save_model(model, output_dir, classifier, C, k)