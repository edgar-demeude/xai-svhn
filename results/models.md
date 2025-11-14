Base CNN : **72.15%** accuracy

```python
class CNN_Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(0.2)

        self.fc = nn.Linear(64*8*8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.pool(x) # (32, 32, 32) -> (32, 16, 16)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.pool(x) # (64, 16, 16) -> (64, 8, 8)
        
        # Flattening of the tensor for the FC layer
        x = x.view(x.size(0), -1)
        
        return self.fc(x)
```

+BatchNormalization : **74.40%** accuracy

```python
class CNN_Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.2)

        self.fc = nn.Linear(64*8*8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.pool(x) # (32, 32, 32) -> (32, 16, 16)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.pool(x) # (64, 16, 16) -> (64, 8, 8)
        
        # Flattening of the tensor for the FC layer
        x = x.view(x.size(0), -1)
        
        return self.fc(x)
```

+Dropout : **74.57%** accuracy

```python
class CNN_Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.1)

        input_features_fc = 64 * 8 * 8
        self.fc = nn.Linear(input_features_fc, 10)
        self.dropout_fc = nn.Dropout(0.4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.pool(x) # (32, 32, 32) -> (32, 16, 16)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.pool(x) # (64, 16, 16) -> (64, 8, 8)
        
        # Flattening of the tensor for the FC layer
        x = x.view(x.size(0), -1)

        x = self.dropout_fc(x)
        
        return self.fc(x)
```

+Layers : **81.41%** accuracy

```python
class CNN_Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # (3, 32, 32) -> (64, 32, 32)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # (64, 32, 32) -> (64, 16, 16)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # (64, 16, 16) -> (128, 16, 16)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(0.1)
        # (128, 16, 16) -> (128, 8, 8)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) # (128, 8, 8) -> (256, 8, 8)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(0.1)
        # (256, 16, 16) -> (256, 4, 4)

        input_features_fc = 256 * 4 * 4
        self.fc = nn.Linear(input_features_fc, 10)
        self.dropout_fc = nn.Dropout(0.4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.pool(x)
        
        # Flattening of the tensor for the FC layer
        x = x.view(x.size(0), -1)

        x = self.dropout_fc(x)
        
        return self.fc(x)
```

+1 Layer : **82.28%** accuracy

```python
class CNN_Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # (3, 32, 32) -> (64, 32, 32)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # (64, 32, 32) -> (64, 16, 16)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # (64, 16, 16) -> (128, 16, 16)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(0.1)
        # (128, 16, 16) -> (128, 8, 8)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) # (128, 8, 8) -> (256, 8, 8)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(0.1)
        # (256, 16, 16) -> (256, 4, 4)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1) # (256, 8, 8) -> (256, 8, 8)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout(0.1)
        # (256, 4, 4) -> (256, 4, 4)

        input_features_fc = 256 * 4 * 4
        self.fc = nn.Linear(input_features_fc, 10)
        self.dropout_fc = nn.Dropout(0.4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        
        # Flattening of the tensor for the FC layer
        x = x.view(x.size(0), -1)

        x = self.dropout_fc(x)
        
        return self.fc(x)
```

+Data augmentation : **83.55%** accuracy

```python
# For training (with Augmentation)
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# For the test (without augmentation, just normalization)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.CIFAR10(root="../data", train=True, download=True, transform=train_transform)
test_data  = datasets.CIFAR10(root="../data", train=False, download=True, transform=test_transform)
```

+Learning Rate Scheduler : **84.12%** accuracy

```python
def train_model(model, dataloader, epochs=20):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs} 🚀 Training", leave=False, colour="green")

        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(preds, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        avg_loss = total_loss / len(dataloader)
        avg_acc = correct / total

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f} - LR: {current_lr:.6f}")
```