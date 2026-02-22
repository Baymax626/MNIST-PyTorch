
import torch
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
