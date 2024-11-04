import os
import random
import csv

def split_dataset(dataset_dir, train_size=500):
    train_data = []
    test_data = []

    # Ensure reproducibility
    random.seed(42)

    # Iterate over each class directory
    class_counter = 0
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        print(f"Processing {class_name} as class {class_counter}")
        if os.path.isdir(class_dir):
            images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
            random.shuffle(images)

            # Split images into train and test sets
            train_images = images[:train_size]
            test_images = images[train_size:]

            # Add to train and test data lists
            for img in train_images:
                train_data.append((img, class_counter))
            for img in test_images:
                test_data.append((img, class_counter))
        class_counter += 1

    return train_data, test_data

def write_csv(data, csv_path):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_path', 'label'])
        writer.writerows(data)

def main():
    dataset_dir = 'miniimagenet'  # Replace with the path to your dataset directory
    train_csv_path = 'train.csv'
    test_csv_path = 'test.csv'

    # Split the dataset
    train_data, test_data = split_dataset(dataset_dir, train_size=500)

    # Write to CSV files
    write_csv(train_data, train_csv_path)
    write_csv(test_data, test_csv_path)

    print(f"Train and test sets generated and saved to {train_csv_path} and {test_csv_path}")

if __name__ == "__main__":
    main()