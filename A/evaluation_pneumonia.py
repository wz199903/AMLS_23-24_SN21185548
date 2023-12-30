import numpy as np
import torch
import torch.nn as nn
from model_pneumonia import ResNet50
from data_preprocessing_pneumonia import data
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO, Evaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = './model_pneumonia.pth'
data_flag = 'pneumoniamnist'

def load_model(model_path):
    model = ResNet50()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, split, test_loader):
    model.eval()
    y_true = torch.tensor([], device=device)
    y_score = torch.tensor([], device=device)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            labels = labels.to(device=device, dtype=torch.float32)
            outputs = torch.sigmoid(outputs)

            y_true = torch.cat((y_true, labels), 0)
            y_score = torch.cat((y_score, outputs), 0)

    y_true = y_true.cpu().numpy()
    y_score = y_score.cpu().detach().numpy()

    evaluator = medmnist.Evaluator(data_flag, split)
    metrics = evaluator.evaluate(y_score)

    print(f'{split.capitalize()} - AUC: {metrics[0]:.3f}, Acc: {metrics[1]:.3f}')
    return y_true, y_score


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()



#def test_different_thresholds(model, test_loader):
    #for threshold in np.arange(0.4, 0.6, 0.05):
        #y_true, y_score = evaluate_model(model, 'split', test_loader)
        # Apply the threshold and evaluate
        #y_pred = (y_score > threshold).astype(int)
        #print(f'Threshold: {threshold}')
        #print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))


def main():
    _, _, test_loader = data(batch_size=64)
    model = load_model(MODEL_PATH)

    y_true, y_score= evaluate_model(model, 'test', test_loader)
    y_pred = (y_score > 0.5).astype(int)

    print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))
    plot_confusion_matrix(y_true, y_pred)
    #test_different_thresholds(model, test_loader)


if __name__ == "__main__":
    main()

