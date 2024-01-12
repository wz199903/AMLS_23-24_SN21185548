import numpy as np
import torch
from model_path import ResNet18, ResNet50, SpecializedResNet50
from data_preprocessing_path import load_data
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import textwrap

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SPECIALISED_MODEL_PATH = 'B/specialised_model_path.pth'

BATCH_SIZE = 64


def main():
    """
    Main function to load data, model, perform evaluation, and plot results
    :return:
    """
    # User input for model selection
    print("Select the base model architecture: ")
    print("1: ResNet18")
    print("2: ResNet50")

    model_class, base_model_options = None, {}
    try:
        model_option = int(input("Enter the number of your choice for the model to be tested: "))
        if model_option == 1:
            model_class = ResNet18
            base_model_options = {"1. ResNet-18 without pretrained weights": "B/model_path_18_no_TL.pth",
                                  "2. ResNet-18 with pretrained weights": "B/model_path_18_V1.pth",
                                  "3. ResNet-18 model trained from train.py": "B/model_path.pth"}
        elif model_option == 2:
            model_class = ResNet50
            base_model_options = {"1. ResNet-50 without pretrained weights": "B/model_path_50_no_TL.pth",
                                  "2. ResNet-50 with pretrained weights V1": "B/model_path_50_V1.pth",
                                  "3. ResNet-50 with pretrained weights V2": "B/model_path_50_V2.pth",
                                  "4. ResNet-50 model trained from train.py": "B/model_path.pth"}
        else:
            print("Invalid choice. Exiting.")
            return
    except ValueError:
        print("Invalid input. Exiting.")
        return

    for key, value in base_model_options.items():
        print(f"{key}: {value}")
    model_version_choice = input("Select the model version: ").strip()

    model_path = base_model_options.get(model_version_choice, list(base_model_options.values())[0])

    print("Choose the evaluation strategy:")
    print("1: Base model predictions only")
    print("2: Ensemble-like predictions with specialised model")
    try:
        strategy_option = int(input("Enter the number of your choice: "))
        if strategy_option not in [1, 2]:
            raise ValueError
    except ValueError:
        print("Invalid input, defaulting to base model predictions only.")
        strategy_option = 1
    _, _, test_loader, _ = load_data(dataset_directory='./Datasets', batch_size=BATCH_SIZE)

    base_model = load_model(model_path, model_class)
    y_true, y_pred, y_scores, auc_scores, accuracy = None, None, None, None, None

    if strategy_option == 1:
        # Evaluate using base model predictions only
        y_true, y_pred, y_scores, auc_scores, accuracy = evaluate_model(base_model, test_loader)
    elif strategy_option == 2:
        # Evaluate using ensemble-like predictions with specialized model
        specialised_model = load_model(SPECIALISED_MODEL_PATH, SpecializedResNet50)
        y_true, y_pred, auc_scores, accuracy = evaluate_combined_model(base_model, specialised_model, test_loader)

    target_names = ['Adipose', 'Background', 'Debris', 'Lymphocytes',
                    'Mucus', 'Smooth Muscle', 'Normal Colon Mucosa',
                    'Cancer-Associated Stroma', 'Colorectal Adenocarcinoma Epithelium']
    print(classification_report(y_true, y_pred, target_names=target_names))
    plot_confusion_matrix(y_true, y_pred, target_names)


def load_model(model_path, model_class):
    """
    Load the pretrained model
    :param model_path: The file path where the pretrained model is.
    :param model_class: The class of the model.
    :return: The loaded model
    """
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


def evaluate_model(model, test_loader, num_classes=9):
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

            y_true.append(labels)
            y_scores.append(outputs)

    y_true = torch.cat(y_true).cpu().numpy()
    y_scores = torch.cat(y_scores).cpu().numpy()

    y_true_one_hot = label_binarize(y_true, classes=range(num_classes))
    auc_scores = roc_auc_score(y_true_one_hot, y_scores, multi_class='ovr')
    y_pred = np.argmax(y_scores, axis=1)
    accuracy = accuracy_score(y_true, y_pred)

    print(f'Test - AUC: {auc_scores:.3f}, Acc: {accuracy:.3f}')
    return y_true, y_pred, y_scores, auc_scores, accuracy


def final_prediction(base_model, specialized_model, inputs, specialized_class_indices=[2, 5, 7]):
    """
    Make the final prediction by combining outputs from base and specialised models
    :param base_model: The base model (ResN
    :param specialized_model:
    :param inputs:
    :param specialized_class_indices:
    :return:
    """

    base_model.eval()
    specialized_model.eval()

    final_preds = []

    # Mapping from the specialised model index to actual class index
    reverse_mapping = {0: 2, 1: 5, 2: 7}

    with torch.no_grad():
        base_probs = torch.softmax(base_model(inputs), dim=1)

        # Iterate over each input
        for i, input_tensor in enumerate(inputs):
            input_tensor = input_tensor.unsqueeze(0)
            base_pred = torch.argmax(base_probs[i]).item()
            final_pred = base_pred

            # If the base model's prediction is one of the specialized classes
            if base_pred in specialized_class_indices:
                # Get specialized model's probabilities for the current input
                specialized_probs = torch.softmax(specialized_model(input_tensor), dim=1).squeeze(0)

                # Map base model's predicted index to specialized model's index
                specialized_index = specialized_class_indices.index(base_pred)

                # Compare softmax probabilities
                if specialized_probs[specialized_index] > base_probs[i][base_pred]:
                    # If specialized model is more confident, use its prediction
                    final_pred = reverse_mapping[specialized_index]

            # Collect the final prediction
            final_preds.append(final_pred)

    return torch.tensor(final_preds)


def evaluate_combined_model(base_model, specialised_model, test_loader, num_classes=9):
    """
    Evaluate the combined model with the test dataset.
    :param base_model: The base model to evaluate
    :param specialised_model: The specialised model for class BACK, MUS, and STR
    :param test_loader: The DataLoader for test data
    :param num_classes:The number of classes in the dataset
    :return: Arrays of true labels, predicted labels, AUC scores, and accuracy
    """
    y_true = []
    y_pred = []
    y_scores = []

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.cpu().numpy()

        # Get probabilities from the base model and the final prediction with the ensemble-like approach
        base_probs = torch.softmax(base_model(inputs), dim=1)
        preds = final_prediction(base_model, specialised_model, inputs).cpu().numpy()

        y_true.extend(labels)
        y_pred.extend(preds)
        y_scores.extend(base_probs.cpu().detach().numpy())

    # Calculate AUC and accuracy
    y_true_one_hot = np.eye(num_classes)[np.array(y_true).reshape(-1)]
    auc_scores = roc_auc_score(y_true_one_hot, np.array(y_scores), multi_class='ovr')
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Test - AUC: {auc_scores:.3f}, Acc: {accuracy:.3f}')

    return y_true, y_pred, auc_scores, accuracy


def plot_confusion_matrix(y_true, y_pred, class_names, wrap_length=14):
    """
    Plot a confusion matrix using seaborn
    :param y_true: Array of true labels
    :param y_pred: Array of predicted labels
    :param class_names: List of class names for labels
    :param wrap_length: Maximum number of characters in a single line for labels
    """
    wrapped_labels = ['\n'.join(textwrap.wrap(label, wrap_length)) for label in class_names]
    cm = confusion_matrix(y_true, y_pred, labels=range(9))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=wrapped_labels, yticklabels=wrapped_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

