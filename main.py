
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

batch_size = 64
lr = 0.001
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"当前使用{device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle= False
)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256,10)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x , y in dataloader:
        x , y = x.to(device), y.to(device)
        #正向传播
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _ , predicted = y_pred.max(y_pred,1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / len(dataloader.dataset)
    return avg_loss, accuracy

def test(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x , y in dataloader:
            x , y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            _ , predicted = y_pred.max(y_pred,1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / len(dataloader.dataset)
    return avg_loss, accuracy
