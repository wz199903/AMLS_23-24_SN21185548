import numpy as np
import torch
from model_path import ResNet18
from data_preprocessing_path import data
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'B/model_path.pth'
#MODEL_PATH = './pretrained_model_path.pth'
BATCH_SIZE = 64


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


def evaluate_model(model, split, test_loader, num_classes=9):
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

            y_true.extend(labels.cpu().numpy())
            y_scores.extend(outputs.cpu().numpy())

    y_true_one_hot = np.eye(num_classes)[np.array(y_true).reshape(-1)]
    y_scores = np.array(y_scores)

    auc_scores = roc_auc_score(y_true_one_hot, y_scores, multi_class='ovr')
    accuracy = accuracy_score(y_true, np.argmax(y_scores, axis=1))

    print(f'{split.capitalize()} - AUC: {auc_scores:.3f}, Acc: {accuracy:.3f}')
    return y_true, y_scores, auc_scores, accuracy


def plot_confusion_matrix(y_true, y_pred):
    """
    Plot a confusion matrix using seaborn
    :param y_true: Array of true labels
    :param y_pred: Array of predicted labels
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(9))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def main():
    """
    Main function to load data, model, perform evaluation, and plot results
    :return:
    """
    _, _, test_loader, _ = data(download_directory='./Datasets', batch_size=BATCH_SIZE)
    model = load_model(MODEL_PATH)
    y_true, y_score, auc_scores, accuracy = evaluate_model(model, 'test', test_loader, num_classes=9)
    y_pred = np.argmax(y_score, axis=1)
    target_names = ['Adipose', 'Background', 'Debris', 'Lymphocytes',
                    'Mucus', 'Smooth Muscle', 'Normal Colon Mucosa',
                    'Cancer-Associated Stroma', 'Colorectal Adenocarcinoma Epithelium']

    print(classification_report(y_true, y_pred, target_names=target_names))
    plot_confusion_matrix(y_true, y_pred)


if __name__ == "__main__":
    main()

