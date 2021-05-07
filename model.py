import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


num_epochs = 10
batch_size = 100
learning_rate = 0.01
num_class = 10
folder = ""     # folder path holding the dataset


transform = transforms.Compose([
    transforms.ToTensor()
    ])

train_data = datasets.ImageFolder(root=folder, transform = transform)
test_data = datasets.ImageFolder(root=folder, transform=transform)

trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


class CNN(nn.Module):
    def __init__(self, numclass):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(in_features=10, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=128)
        self.fc = nn.Linear(in_features=128 , out_features=num_class)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        # out = out.reshape(out.size(0), -1)
        out = F.tanh(self.fc(out))

        return out


model = CNN(num_class)

optimizer = optim.Adam(model.parameters(), learning_rate)
criterion = nn.BCELoss()

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        # forward pass
        output = model(images)
        loss = criterion(output, labels)

        # back pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epochs {epoch + 1}/{num_epochs}  Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(
        100 * correct / total))


