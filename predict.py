import torch
from torchvision import datasets, transforms

# Aynı transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# Test datasını yükle
test_dataset  = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# Eğittiğin modeli tekrar oluşturalım
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()

# Eğitilmiş modeli mnist.py içinden kaydedelim demedik, o yüzden
# şimdilik predict için mnist.py’de eğitim kısmının en sonuna ŞUNU ekle
# torch.save(model.state_dict(), "mnist_model.pth")
#
# Eğer eklediysen burayı çalıştıracağız ↓

model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Rastgele bir görüntü alalım
image, label = test_dataset[0]

with torch.no_grad():
    output = model(image.unsqueeze(0))
    predicted = torch.argmax(output, 1).item()

print("Gerçek Label:", label)
print("Model Tahmini:", predicted)


