"""
Early Diagnosis of Alzheimer's using the Weighted Probability Based Ensemble Method
Hamse Elmi - U551835 - 2023232
Bachelor Thesis
"""

# --- Imports ---
# Importing essential libraries for data handling, model definitions, and evaluation
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import (roc_curve, auc, confusion_matrix, classification_report,
                             f1_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from skimage.measure import shannon_entropy
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Preprocessing NIfTI Files ---

# Dictionary mapping each category to its corresponding data directory
data_dirs = {
    'ad': '/home/u551835/Data/Alzheimer/AD',
    'mci': '/home/u551835/Data/Alzheimer/MCI',
    'n': '/home/u551835/Data/Alzheimer/N'
}

# Initialize a dictionary to hold lists of processed slices for each category
slices_by_category = {'ad': [], 'mci': [], 'n': []}

def preprocess_nifti(data):
    """
    Preprocesses a NIfTI volume by:
    1. Handling 4D data by selecting the first 3D volume.
    2. Removing some slices from the start and end if the volume has >60 slices.
    3. Normalizing pixel intensities to [0,1].
    4. Computing variance across slices, selecting slices above the 20th percentile variance.
    5. Resizing each selected slice to 256x256.
    
    Returns:
        A list of resized 2D slices (as numpy arrays).
    """
    # If the data is 4D, we only take the first volume
    if data.ndim == 4:
        data = data[..., 0]
    
    # If there are more than 60 slices, remove 20 slices from start and end
    if data.shape[2] > 60:
        data = data[:, :, 20:-20]
    
    # Normalize the data if possible (avoid divide by zero)
    data = (data - data.min()) / (data.max() - data.min()) if data.max() > data.min() else data
    
    # Calculate variance for each slice in the 3rd dimension
    slice_variances = [np.var(data[:, :, i]) for i in range(data.shape[2])]
    
    # Use the 20th percentile of variance as threshold
    threshold = np.percentile(slice_variances, 20)
    
    # Select the indices of slices whose variance is above the threshold
    brain_slices = [i for i in range(data.shape[2]) if slice_variances[i] > threshold]
    
    # Resize the selected slices to 256x256
    resized_slices = [resize(data[:, :, i], (256, 256), anti_aliasing=True) for i in brain_slices]
    
    return resized_slices

def process_directory(category, directory):
    """
    Reads NIfTI files from a directory, preprocesses them, calculates slice entropy,
    and keeps the top 20 slices by entropy. Stores these slices in slices_by_category.
    
    Args:
        category (str): 'ad', 'mci', or 'n'.
        directory (str): Path to the directory containing NIfTI files.
    """
    for filename in os.listdir(directory):
        # Check for NIfTI file extensions
        if filename.endswith(('.nii', '.nii.gz')):
            file_path = os.path.join(directory, filename)
            img = nib.load(file_path)      # Load the NIfTI file
            data = img.get_fdata()         # Extract the image data as a NumPy array
            
            # Preprocess the data to get a list of slices
            preprocessed_slices = preprocess_nifti(data)
            
            # Calculate the Shannon entropy of each slice
            entropies = [shannon_entropy(slice_) for slice_ in preprocessed_slices]
            
            # Get indices of the top 20 slices based on entropy
            top_slices = sorted(range(len(entropies)), key=lambda x: entropies[x], reverse=True)[:20]
            
            # Extend the list in slices_by_category with these top 20 slices
            slices_by_category[category].extend([preprocessed_slices[i] for i in top_slices])

# Process all directories (AD, MCI, Normal) and populate slices_by_category
for category, directory in data_dirs.items():
    process_directory(category, directory)

# Combine all processed data and corresponding labels into lists
all_data = []
all_labels = []
category_to_label = {'ad': 0, 'mci': 1, 'n': 2}

# Populate all_data with the slices and all_labels with numeric labels
for category, slices in slices_by_category.items():
    all_data.extend(slices)
    all_labels.extend([category_to_label[category]] * len(slices))

# Convert the lists to numpy arrays for further processing
all_data = np.array(all_data)
all_labels = np.array(all_labels)

# Split the data into training and temp sets (80% train, 20% temp)
train_data, temp_data, train_labels, temp_labels = train_test_split(
    all_data, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

# Then split the temp set equally into validation and test sets (10% val, 10% test overall)
val_data, test_data, val_labels, test_labels = train_test_split(
    temp_data, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

# --- Step 2: Dataset and DataLoader ---

class PETDataset(Dataset):
    """
    Custom PyTorch Dataset for PET slices.
    """
    def __init__(self, data, labels):
        """
        Args:
            data (np.array): Array of slices with shape (num_slices, 256, 256).
            labels (np.array): Corresponding labels of shape (num_slices,).
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        # Return total number of slices
        return len(self.data)

    def __getitem__(self, idx):
        # Convert the slice to a torch tensor and add a channel dimension
        image = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)
        # Convert label to a torch tensor of type long
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# Create dataset objects for train, validation, and test sets
train_dataset = PETDataset(train_data, train_labels)
val_dataset = PETDataset(val_data, val_labels)
test_dataset = PETDataset(test_data, test_labels)

# Create DataLoader objects to load data in batches
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- Step 3: Model Definitions ---

class VGG19Model(nn.Module):
    """
    Custom VGG19-based model for 1-channel (grayscale) input and reduced layers.
    """
    def __init__(self, num_classes=3):
        super(VGG19Model, self).__init__()
        # Load a VGG19 base model (here weights=None, can be changed to pretrained if desired)
        self.base_model = models.vgg19(weights=None)
        
        # Modify the first convolutional layer to accept 1 input channel
        self.base_model.features[0] = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # Restrict the features to the first 6 layers of VGG19
        self.base_model.features = nn.Sequential(
            *list(self.base_model.features.children())[:6]
        )

        # Add custom layers after the truncated VGG19 feature extractor
        self.additional_layers = nn.Sequential(
            nn.BatchNorm2d(128),               # BatchNorm for the last conv output channels
            nn.Flatten(),                      # Flatten for fully-connected layer
            nn.Linear(128 * 128 * 128, 64),    # FC layer
            nn.BatchNorm1d(64),                # BatchNorm for FC layer
            nn.Dropout(0.5),                   # Dropout for regularization
            nn.ReLU(),                         # Activation
            nn.Linear(64, 32),                 # Second FC layer
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(32, num_classes)         # Final output layer
        )

    def forward(self, x):
        # Pass the input through the truncated VGG19 features
        x = self.base_model.features(x)
        # Pass the result through the additional classification layers
        x = self.additional_layers(x)
        return x

class DenseNet201Model(nn.Module):
    """
    Custom DenseNet201-based model for 1-channel input.
    """
    def __init__(self, num_classes=3):
        super(DenseNet201Model, self).__init__()
        # Load a DenseNet201 model (weights=None for no pretrained weights)
        self.base_model = models.densenet201(weights=None)
        
        # Modify the initial convolution to accept 1 channel
        self.base_model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.base_model.features.conv0.weight, mode="fan_out", nonlinearity="relu")
        
        # Modify the final classifier layer to match the number of classes
        num_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

class ResNet50Model(nn.Module):
    """
    Custom ResNet50-based model for 1-channel input.
    """
    def __init__(self, num_classes=3):
        super(ResNet50Model, self).__init__()
        # Load a ResNet50 model (weights=None for no pretrained weights)
        self.base_model = models.resnet50(weights=None)
        
        # Modify the initial convolution to accept 1 channel
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.base_model.conv1.weight, mode="fan_out", nonlinearity="relu")
        
        # Modify the final FC layer to match the number of classes
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# Dictionary that maps model names to corresponding model class instances
models_dict = {
    "DenseNet201": DenseNet201Model(num_classes=3),
    "ResNet50": ResNet50Model(num_classes=3),
    "VGG19": VGG19Model(num_classes=3)
}

# --- Step 4: Training and Evaluation ---

# Define a loss function for multi-class classification
criterion = nn.CrossEntropyLoss()

# Detect GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_evaluate(model, train_loader, val_loader, num_epochs, lr):
    """
    Train a given model using the specified dataloaders, number of epochs, and learning rate.
    Evaluate after each epoch on the validation set and keep track of the best accuracy.
    
    Args:
        model: A PyTorch model instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        num_epochs (int): Number of epochs to train.
        lr (float): Learning rate for the optimizer.
    
    Returns:
        float: The best validation accuracy achieved during training.
    """
    model.to(device)  # Move model to the chosen device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam optimizer
    
    best_accuracy = 0  # Track the best validation accuracy
    for epoch in range(num_epochs):
        # --- Training ---
        model.train()  # Set model to training mode
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Reset gradient
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Optimize
            optimizer.step()
        
        # --- Validation ---
        model.eval()  # Set model to evaluation mode
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)   # Get class index with highest probability
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_accuracy = correct / total
        best_accuracy = max(best_accuracy, val_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}")
    
    return best_accuracy

# --- Step 5: Confusion Matrix and ROC Curve ---

def plot_confusion_matrix(true_labels, pred_labels, model_name):
    """
    Plots and saves a confusion matrix for a given set of true and predicted labels.
    """
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Plot the confusion matrix using Seaborn heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=['AD', 'MCI', 'N'], yticklabels=['AD', 'MCI', 'N'])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(f"confusion_matrix_{model_name}.png")
    plt.show()

def plot_roc_curve(true_labels, softmax_outputs, model_name):
    """
    Plots and saves a multiclass ROC curve using the softmax outputs of the model.
    """
    # Binarize the true labels for multiclass ROC
    binarized_labels = label_binarize(true_labels, classes=[0, 1, 2])
    
    # Plot ROC curve for each class
    for i in range(3):
        fpr, tpr, _ = roc_curve(binarized_labels[:, i], softmax_outputs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
    
    # Diagonal line for random chance
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(f"roc_curve_{model_name}.png")
    plt.show()

# --- Step 6: WPBM Ensemble ---
# Section provided by Fathi
def wpbm_prediction(test_loader, models, weights):
    """
    Weighted Probability-Based Majority (WPBM) prediction.
    Aggregates predictions from multiple models according to specified weights.
    
    Args:
        test_loader: DataLoader for the test set.
        models (list): List of trained model instances.
        weights (list): List of weights for each model's predictions.
    
    Returns:
        tuple of (final_predictions, weighted_probabilities)
    """
    predictions = []
    
    # Generate softmax probability predictions for each model
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                # Get softmax probabilities for each batch
                preds.append(torch.softmax(model(images), dim=1).cpu().numpy())
        # Stack predictions along axis 0
        predictions.append(np.vstack(preds))
    
    # Convert to NumPy array for easy math operations
    predictions = np.array(predictions)
    
    # Apply the weights to each model's predictions and sum
    weighted_preds = np.sum([pred * weights[i] for i, pred in enumerate(predictions)], axis=0)
    
    # Argmax across classes for final prediction
    return np.argmax(weighted_preds, axis=1), weighted_preds

# --- Step 7: Training Models and Ensemble ---

# Dictionaries to store trained models and their best validation accuracies
trained_models = {}
model_accuracies = {}

# Train and evaluate each model in models_dict
for model_name, model in models_dict.items():
    print(f"Training {model_name}...")
    accuracy = train_and_evaluate(model, train_loader, val_loader, num_epochs=350, lr=0.001)
    trained_models[model_name] = model
    model_accuracies[model_name] = accuracy
    print(f"{model_name} Best Validation Accuracy: {accuracy:.4f}")

# Compute ensemble weights based on each model's validation accuracy
total_accuracy = sum(model_accuracies.values())
ensemble_weights = [model_accuracies[model_name] / total_accuracy for model_name in models_dict.keys()]

# Get WPBM ensemble predictions on the test set
ensemble_predictions, ensemble_softmax = wpbm_prediction(
    test_loader, list(trained_models.values()), ensemble_weights
)

# Evaluate ensemble model (collect true labels and WPBM outputs)
true_labels = []
softmax_outputs = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        # Get softmax outputs from each model
        outputs = [torch.softmax(model(images), dim=1) for model in trained_models.values()]
        # Combine outputs using ensemble weights
        combined_outputs = sum(w * output for w, output in zip(ensemble_weights, outputs))
        
        # Store combined outputs and true labels
        softmax_outputs.extend(combined_outputs.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

softmax_outputs = np.array(softmax_outputs)
true_labels = np.array(true_labels)

# Plot confusion matrix and ROC curve for the WPBM Ensemble
plot_confusion_matrix(true_labels, ensemble_predictions, "WPBM Ensemble")
plot_roc_curve(true_labels, softmax_outputs, "WPBM Ensemble")

# Save the trained models
for model_name, model in trained_models.items():
    torch.save(model.state_dict(), f"{model_name}_final.pth")

# --- Step 8: Performance Evaluation ---

def evaluate_metrics(true_labels, pred_labels, softmax_outputs, model_name):
    """
    Evaluate performance metrics (Classification Report, AUC, F1 Score, Sensitivity, Specificity)
    and plot a confusion matrix heatmap.
    
    Args:
        true_labels (np.array): Ground truth labels.
        pred_labels (np.array): Model-predicted labels.
        softmax_outputs (np.array): Softmax probabilities for each class.
        model_name (str): Name of the model (used for labeling plots/saving).
    
    Returns:
        dict: Dictionary with performance metrics.
    """
    # Print classification report (precision, recall, F1-score)
    report = classification_report(true_labels, pred_labels,
                                   target_names=['AD', 'MCI', 'N'],
                                   output_dict=True, zero_division=0)
    print(f"Classification Report for {model_name}:")
    print(report)
    
    # Calculate AUC in a one-vs-rest manner
    auc_score = roc_auc_score(label_binarize(true_labels, classes=[0, 1, 2]),
                              softmax_outputs, multi_class='ovr')
    print(f"AUC for {model_name}: {auc_score:.2f}")
    
    # Calculate weighted F1 score
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    print(f"F1 Score for {model_name}: {f1:.2f}")
    
    # Calculate sensitivity and specificity from confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # For a 3-class problem, sensitivity/specificity can be calculated per class.
    # Here, an example is shown for class 1 vs. rest.
    # For more precise class-wise measures, each class should be computed separately.
    # The code below is an example for class 'MCI' (index 1).
    
    # Sensitivity (Recall) for class 1
    sensitivity = (cm[1, 1] / (cm[1, 1] + cm[1, 0])) if (cm[1, 1] + cm[1, 0]) != 0 else 0
    # Specificity for class 1 (we look at the 'not class 1' block)
    specificity = (cm[0, 0] / (cm[0, 0] + cm[0, 1])) if (cm[0, 0] + cm[0, 1]) != 0 else 0
    print(f"Sensitivity for {model_name}: {sensitivity:.2f}")
    print(f"Specificity for {model_name}: {specificity:.2f}")
    
    # Plot confusion matrix heatmap for the model
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['AD', 'MCI', 'N'], yticklabels=['AD', 'MCI', 'N'])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(f"confusion_matrix_{model_name}.png")
    plt.show()
    
    # Return metrics in a dictionary for record-keeping
    return {
        'Model': model_name,
        'Accuracy': (cm.diagonal().sum() / cm.sum()) * 100,  # Overall accuracy in percentage
        'F1 Score': f1,
        'AUC': auc_score,
        'Sensitivity': sensitivity,
        'Specificity': specificity
    }

# Evaluate individual trained models on the test set
performance_metrics = []

for model_name, model in trained_models.items():
    print(f"Evaluating {model_name}...")
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    all_softmax_outputs = []
    
    # Disable gradient calculation
    with torch.no_grad():
        # Iterate over test data batches
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            # Convert logits to softmax probabilities
            softmax_batch = torch.softmax(outputs, dim=1).cpu().numpy()
            # Predict labels based on highest probability
            pred_labels = np.argmax(softmax_batch, axis=1)
            
            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(pred_labels)
            all_softmax_outputs.extend(softmax_batch)
    
    # Evaluate and store metrics
    metrics = evaluate_metrics(
        np.array(all_true_labels),
        np.array(all_pred_labels),
        np.array(all_softmax_outputs),
        model_name
    )
    performance_metrics.append(metrics)

# Evaluate the WPBM Ensemble predictions
print("Evaluating WPBM Ensemble...")
all_true_labels = []

# Get ensemble predictions and softmax outputs
ensemble_predictions, ensemble_softmax = wpbm_prediction(
    test_loader, list(trained_models.values()), ensemble_weights
)

# Collect true labels
with torch.no_grad():
    for _, labels in test_loader:
        all_true_labels.extend(labels.numpy())

# Evaluate ensemble model
ensemble_metrics = evaluate_metrics(
    np.array(all_true_labels),
    ensemble_predictions,
    ensemble_softmax,
    "WPBM Ensemble"
)
performance_metrics.append(ensemble_metrics)

# --- Step 9: Performance Comparison Bar Graph ---

def plot_performance_bar_chart(metrics):
    """
    Plots a bar chart comparing model performances (Accuracy, F1, AUC) for each model.
    """
    model_names = [m['Model'] for m in metrics]
    accuracies = [m['Accuracy'] for m in metrics]
    f1_scores = [m['F1 Score'] for m in metrics]
    auc_scores = [m['AUC'] for m in metrics]
    
    x = np.arange(len(model_names))
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the three metrics side by side
    ax.bar(x - width, accuracies, width, label='Accuracy')
    ax.bar(x, f1_scores, width, label='F1 Score')
    ax.bar(x + width, auc_scores, width, label='AUC')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    # Optionally add the exact values on top of each bar
    for i in range(len(model_names)):
        ax.text(x[i] - width, accuracies[i] + 0.5, f'{accuracies[i]:.2f}%', ha='center')
        ax.text(x[i], f1_scores[i] + 0.5, f'{f1_scores[i]:.2f}', ha='center')
        ax.text(x[i] + width, auc_scores[i] + 0.5, f'{auc_scores[i]:.2f}', ha='center')

    plt.savefig('model_performance_comparison.png')
    plt.show()

# Plot the performance comparison bar chart
plot_performance_bar_chart(performance_metrics)

# --- Step 10: Save Metrics to CSV ---
import pandas as pd

# Create a DataFrame from the collected performance metrics and save to CSV
metrics_df = pd.DataFrame(performance_metrics)
metrics_df.to_csv('model_performance_metrics.csv', index=False)
print("Performance metrics saved to 'model_performance_metrics.csv'.")
