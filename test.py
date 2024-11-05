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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

def show_args(args):
    print(f'---------- Arguments -----------')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print(f'-------------------------------')

def load_model(path):
    model = pickle.load(open(path, 'rb'))
    return model

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

if __name__ == '__main__':

    # args
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default='weights/svm_C_1.pkl')
    args = parser.parse_args()

    # show arguments
    show_args(args)
    model_path = args.model_path

    # Load the classifier
    model = load_model(model_path)

    # Test the classifier
    test_dataset = ImageDataset(annotaion='test')
    print(f'Test dataset size: {len(test_dataset)}')
    accuracy, precision, recall, f1 = test_classifier(model, test_dataset)
    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')