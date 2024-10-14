import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置超参数
num_classes = 10
batch_size = 32
learning_rate = 0.001
epochs = 100

# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
])
 

train_dataset = datasets.MNIST(root='data', train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义验证集
val_dataset = datasets.MNIST(root='data', train=False, download=False, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 提前停止的参数
patience = 5  # 容忍的epoch数
best_loss = float('inf')
trigger_times = 0

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练循环
idx=1
while True:  # 不限制epochs，使用无限循环
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 打印每个epoch的平均训练损失
    average_loss = running_loss / len(train_loader)
    print(f"{idx},Average Training Loss: {average_loss:.4f}", end="")
    idx+=1
    # 验证集评测
    model.eval()  # 设置模型为评估模式
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    average_val_loss = val_loss / len(val_loader)
    print(f"  Validation Loss: {average_val_loss:.4f}")

    # 提前停止逻辑
    if average_val_loss < best_loss:
        best_loss = average_val_loss
        trigger_times = 0  # 重置触发次数
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered")
            break
print("Training complete")
