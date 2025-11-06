import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve, auc
import numpy as np
import pandas as pd
import time
import seaborn as sns
from collections import defaultdict
import random
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

subset_size = 70000
num_classes = 10
per_class = subset_size // num_classes

class_indices = defaultdict(list)
for idx, (_, label) in enumerate(mnist_dataset):
    if len(class_indices[label]) < per_class:
        class_indices[label].append(idx)
    if all(len(v) == per_class for v in class_indices.values()):
        break

train_indices = []
test_indices = []

for cls in range(num_classes):
    cls_idx = class_indices[cls]
    np.random.shuffle(cls_idx)
    split_point = per_class // 5
    test_indices.extend(cls_idx[:split_point])
    train_indices.extend(cls_idx[split_point:])

train_dataset = Subset(mnist_dataset, train_indices)
test_dataset = Subset(mnist_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class CustomFunction(nn.Module):
    def forward(self, x):
        common = x * torch.tanh(x / (1 + torch.exp(-0.5 * x)) + 0.5)
        return torch.where(x > 0, common, common * torch.exp(0.5 * x))

activation_functions = {
    "ReLU": nn.ReLU,
    "ELU": nn.ELU,
    "Swish": Swish,
    "Mish": Mish,
    "GELU": nn.GELU,
    "Custom": CustomFunction
}

def build_lenet(activation):
    return nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5),
        activation(),
        nn.AvgPool2d(2),
        nn.Conv2d(6, 16, kernel_size=5),
        activation(),
        nn.AvgPool2d(2),
        nn.Flatten(),
        nn.Linear(16*5*5, 120),
        activation(),
        nn.Linear(120, 84),
        activation(),
        nn.Linear(84, 10)
    )

def build_alexnet(activation):
    model = models.alexnet()
    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    model.features[2] = nn.MaxPool2d(kernel_size=2, stride=2)
    model.classifier[6] = nn.Linear(4096, 10)
    return model

def build_vggnet(activation):
    model = models.vgg11()
    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
    model.classifier[6] = nn.Linear(4096, 10)
    return model

def build_googlenet(activation):
    model = models.googlenet(aux_logits=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(1024, 10)
    return model

def build_resnet18(activation):
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 10)
    return model

def build_mobilenet(activation):
    model = models.mobilenet_v2()
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    return model

def build_densenet(activation):
    model = models.densenet121()
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Linear(model.classifier.in_features, 10)
    return model

def build_efficientnetb0(activation):
    model = models.efficientnet_b0()
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    return model

architectures = {
    "LeNet": build_lenet,
    "AlexNet": build_alexnet,
    "VGGNet": build_vggnet,
    "GoogLeNet": build_googlenet,
    "ResNet18": build_resnet18,
    "MobileNetV2": build_mobilenet,
    "DenseNet121": build_densenet,
    "EfficientNetB0": build_efficientnetb0
}

def train_model(model, train_loader, test_loader, device, epochs=100):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_acc_list, test_acc_list, train_loss_list, test_loss_list = [], [], [], []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        correct, total, epoch_loss = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = correct / total
        train_loss = epoch_loss / total
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        test_acc, test_loss, _, _ = evaluate_model(model, test_loader, device, criterion)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        print(f"Epoch [{epoch+1}/{epochs}] Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    time_taken = time.time() - start_time
    return model, train_acc_list, test_acc_list, train_loss_list, test_loss_list, time_taken

def evaluate_model(model, data_loader, device, criterion=None):
    model.eval()
    correct, total, total_loss = 0, 0, 0
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            if criterion:
                total_loss += criterion(outputs, labels).item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    acc = correct / total
    avg_loss = total_loss / total if criterion else 0
    return acc, avg_loss, np.array(all_labels), np.array(all_probs)

def compute_metrics(labels, probs):
    preds = np.argmax(probs, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    try:
        auc_macro = roc_auc_score(labels, probs, multi_class='ovo', average='macro')
    except:
        auc_macro = None
    return precision, recall, f1, auc_macro

def plot_confusion(labels, preds, title="Confusion Matrix"):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_curves(train_values, test_values, ylabel, title):
    plt.figure()
    plt.plot(train_values, label='Train')
    plt.plot(test_values, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

    df = pd.DataFrame({
        'Epoch': np.arange(1, len(train_values)+1),
        f'Train_{ylabel}': np.round(train_values, 4),
        f'Test_{ylabel}': np.round(test_values, 4)
    })

    print("\n" + "="*50)
    print(f"{title} (Numerical Data)")
    print("="*50)
    print(df.to_csv(sep="\t", index=False))
    print("="*50 + "\n")

def plot_macro_roc(labels, probs, n_classes=10):
    fpr_common = np.linspace(0, 1, 101)
    tpr_interpolated = []

    for i in range(n_classes):
        y_true = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true, probs[:, i])
        if fpr[0] > 0:
            fpr = np.insert(fpr, 0, 0.0)
            tpr = np.insert(tpr, 0, 0.0)
        if fpr[-1] < 1:
            fpr = np.append(fpr, 1.0)
            tpr = np.append(tpr, 1.0)
        tpr_interp = np.interp(fpr_common, fpr, tpr)
        tpr_interpolated.append(tpr_interp)

    tpr_macro = np.mean(tpr_interpolated, axis=0)
    auc_macro = auc(fpr_common, tpr_macro)

    plt.figure()
    plt.plot(fpr_common, tpr_macro, color='b', label=f'Macro-Averaged ROC (AUC={auc_macro:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Macro-Averaged ROC Curve')
    plt.legend()
    plt.show()

    df = pd.DataFrame({
        'FPR': np.round(fpr_common, 2),
        'TPR': np.round(tpr_macro, 6)
    })

    print("\n" + "="*50)
    print("Macro-Averaged ROC (Numerical Data)")
    print("="*50)
    print(df.to_csv(sep="\t", index=False))
    print("="*50 + "\n")
    return fpr_common, tpr_macro, auc_macro

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = []

for arch_name, arch_fn in architectures.items():
    for act_name, act_fn in activation_functions.items():
        print(f"\nTraining {arch_name} with {act_name} activation...")
        model = arch_fn(act_fn)
        model, train_acc_list, test_acc_list, train_loss_list, test_loss_list, time_taken = train_model(
            model, train_loader, test_loader, device, epochs=100
        )

        train_acc = train_acc_list[-1]
        test_acc = test_acc_list[-1]
        _, _, train_labels, train_probs = evaluate_model(model, train_loader, device)
        _, _, test_labels, test_probs = evaluate_model(model, test_loader, device)
        precision, recall, f1, auc_macro = compute_metrics(test_labels, test_probs)

        print(f"\nFINAL RESULTS for {arch_name} with {act_name}:")
        print(f"Time Taken: {time_taken:.2f}s")
        print(f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        print(f"Weighted Precision: {precision:.4f} | Weighted Recall: {recall:.4f} | Weighted F1 Score: {f1:.4f}")

        plot_confusion(train_labels, np.argmax(train_probs, axis=1), title=f"{arch_name} with {act_name} (Train Confusion Matrix)")
        plot_confusion(test_labels, np.argmax(test_probs, axis=1), title=f"{arch_name} with {act_name} (Test Confusion Matrix)")
        plot_curves(train_acc_list, test_acc_list, ylabel="Accuracy", title=f"{arch_name} with {act_name} (Accuracy Curve)")
        plot_curves(train_loss_list, test_loss_list, ylabel="Loss", title=f"{arch_name} with {act_name} (Loss Curve)")
        fpr_macro, tpr_macro, auc_macro = plot_macro_roc(test_labels, test_probs)
        results.append((arch_name, act_name, train_acc, test_acc, precision, recall, f1, auc_macro, time_taken))

# Happy Coding!