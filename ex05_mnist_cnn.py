import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MnistCNN(nn.Module):
    def __init__(self, use_batchnorm=False, use_dropout=False):
        super(MnistCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(32) if use_batchnorm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()

        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def build_optimizer(model, opt_name="adam"):
    if opt_name == "adam":
        return optim.Adam(model.parameters(), lr=0.001)
    elif opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    return optim.Adam(model.parameters(), lr=0.001)

def build_scheduler(optimizer, sched_name="none", epochs=10):
    if sched_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    return None

def get_loaders(augment=False, batch_size=256):
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, shear=10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomErasing(p=0.1)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = torchvision.datasets.MNIST(root="./data", train=True,
                                             download=True, transform=train_transform)
    test_data = torchvision.datasets.MNIST(root="./data", train=False,
                                            download=True, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()
    return 100.0 * correct / total

def train_model(model, train_loader, test_loader, optimizer, scheduler, epochs, name="Model"):
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    start_time = time.time()
    print(f"\n--- Training: {name} ---")
    print(f"{'Epoch':<8}{'Train Loss':<15}{'Train Acc%':<15}{'Test Acc%':<15}{'GPU Mem MB':<12}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item()
            predicted = outputs.argmax(dim=1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()

        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        test_acc = evaluate(model, test_loader)

        if torch.cuda.is_available():
            mem_mb = torch.cuda.memory_allocated() / 1e6
        else:
            mem_mb = 0

        print(f"{epoch+1:<8}{avg_loss:<15.4f}{train_acc:<15.2f}{test_acc:<15.2f}{mem_mb:<12.1f}")

    elapsed = time.time() - start_time
    final_acc = evaluate(model, test_loader)
    return final_acc, elapsed

def run_ablation():
    print("\n========== ABLATION STUDY ==========")
    train_loader, test_loader = get_loaders(augment=False)

    configs = [
        ("Baseline Adam no BN/DO", False, False, "adam", "none"),
        ("Adam + BatchNorm",        True,  False, "adam", "none"),
        ("Adam + Dropout",          False, True,  "adam", "none"),
        ("SGD+Momentum+Cosine",     False, False, "sgd",  "cosine"),
    ]

    results = []
    for name, bn, do, opt_name, sched_name in configs:
        model = MnistCNN(use_batchnorm=bn, use_dropout=do)
        optimizer = build_optimizer(model, opt_name)
        scheduler = build_scheduler(optimizer, sched_name, epochs=5)
        acc, elapsed = train_model(model, train_loader, test_loader, optimizer, scheduler, 5, name)
        results.append((name, acc, elapsed))

    print("\n--- Ablation Results ---")
    print(f"{'Config':<35}{'Test Acc%':<15}{'Time (s)':<10}")
    for name, acc, elapsed in results:
        print(f"{name:<35}{acc:<15.2f}{elapsed:<10.1f}")

def run_augmentation_test():
    print("\n========== DATA AUGMENTATION TEST ==========")

    train_loader_no_aug, test_loader = get_loaders(augment=False)
    model1 = MnistCNN()
    opt1 = build_optimizer(model1, "adam")
    acc_no_aug, _ = train_model(model1, train_loader_no_aug, test_loader, opt1, None, 10, "No Augmentation")

    train_loader_aug, _ = get_loaders(augment=True)
    model2 = MnistCNN()
    opt2 = build_optimizer(model2, "adam")
    acc_aug, _ = train_model(model2, train_loader_aug, test_loader, opt2, None, 10, "With Augmentation")

    print(f"\nNo augmentation final accuracy: {acc_no_aug:.2f}%")
    print(f"With augmentation final accuracy: {acc_aug:.2f}%")

def run_amp_training():
    if not torch.cuda.is_available():
        print("AMP requires CUDA, skipping.")
        return

    print("\n========== AMP TRAINING ==========")
    train_loader, test_loader = get_loaders()
    model = MnistCNN().to(device)
    optimizer = build_optimizer(model, "adam")
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(5):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    elapsed = time.time() - start
    acc = evaluate(model, test_loader)
    mem_mb = torch.cuda.memory_allocated() / 1e6
    print(f"AMP 5 epochs: {elapsed:.1f}s, acc: {acc:.2f}%, peak mem: {mem_mb:.1f} MB")

if __name__ == "__main__":
    run_ablation()
    run_augmentation_test()
    run_amp_training()
