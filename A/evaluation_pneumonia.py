import numpy as np
import torch
from model_pneumonia import ResNet18
from data_preprocessing_pneumonia import load_data
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Choose the model (Default name of the trained model: model_pneumonia.pth)
#MODEL_PATH = 'A/model_pneumonia.pth'
MODEL_PATH = 'A/model_pneumonia_no_TL.pth'
# MODEL_PATH = 'A/model_pneumonia_TL_V1.pth'

# Hyperparameter
BATCH_SIZE = 32


def load_model(model_path):
    """
    Load the pretrained model
    :param model_path: The file path where the pretrained model is.
    :return: The loaded model
    """
    model = ResNet18()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


def evaluate_model(model, split, test_loader):
    """
    Evaluate the model using the provided test_loader.
    :param model: The neural network model to evaluate.
    :param split: The split type, typically 'test' in evaluation.
    :param test_loader: The DataLoader for test data.
    :return: Arrays of true labels and predicted scores
    """
    model.eval()

    y_true = []
    y_scores = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze()

            y_true.extend(labels.cpu().numpy())
            y_scores.extend(outputs.cpu().numpy())

    # Convert lists of labels and scores to numpy arrays for further analysis
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Convert scores to binary predictions and calculate AUC and accuracy
    predictions = (y_scores > 0.5).astype(int)
    auc_score = roc_auc_score(y_true, y_scores)
    accuracy = accuracy_score(y_true, predictions)

    print(f'{split.capitalize()} - AUC: {auc_score:.3f}, Acc: {accuracy:.3f}')
    return y_true, y_scores, auc_score, accuracy


def plot_confusion_matrix(y_true, y_pred):
    """
    Plot a confusion matrix using seaborn
    :param y_true: Array of true labels
    :param y_pred: Array of predicted labels
    """
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal',
                                                                                                         'Pneumonia'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to load data, model, perform evaluation, and plot results
    """
    print("Enter the model path:")
    print("1: Model saved from the training phase")
    print("2: Pretrained model without transfer learning weights")
    print("3: Pretrained model with transfer learning weights")
    model_choice = input("Enter your choice (1, 2, 3) or custom path: ").strip()

    model_paths = {
        "1": 'A/model_pneumonia.pth',
        "2": 'A/model_pneumonia_no_TL.pth',
        "3": 'A/model_pneumonia_TL_V1.pth'
    }
    model_path = model_paths.get(model_choice, model_choice) if model_choice else 'A/model_pneumonia_no_TL.pth'
    _, _, test_loader, _ = load_data(dataset_directory='./Datasets', batch_size=BATCH_SIZE)
    model = load_model(model_path)

    y_true, y_score, auc_score, accuracy = evaluate_model(model, 'test', test_loader)
    y_pred = (y_score > 0.5).astype(int)

    print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))
    plot_confusion_matrix(y_true, y_pred)


if __name__ == "__main__":
    main()
