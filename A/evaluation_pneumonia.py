import numpy as np
import torch
from model_pneumonia import ResNet18
from data_preprocessing_pneumonia import data
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = './model_pneumonia_38.pth'
#MODEL_PATH = './pretrained_model_pneumonia.pth'
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

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    auc_score = roc_auc_score(y_true, y_scores)
    predictions = (y_scores > 0.5).astype(int)
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
    plt.show()


def main():
    """
    Main function to load data, model, perform evaluation, and plot results
    :return:
    """
    _, _, test_loader, _ = data(dataset_directory='../Datasets', batch_size=BATCH_SIZE)
    model = load_model(MODEL_PATH)

    y_true, y_score, auc_score, accuracy = evaluate_model(model, 'test', test_loader)
    y_pred = (y_score > 0.5).astype(int)

    print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))
    plot_confusion_matrix(y_true, y_pred)


if __name__ == "__main__":
    main()
