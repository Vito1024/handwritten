import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.LeNet = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10),
        )

    def forward(self, X):
        for layer in self.LeNet:
            X = layer(X)
        return F.log_softmax(X, dim=1)


if __name__ == '__main__':

    net = LeNet().to(device)

    # train
    train_loader = torchvision.datasets.MNIST(
        root='../',
        download=True,
        transform=torchvision.transforms.ToTensor(),
        train=True,
    )
    train_loader = torch.utils.data.DataLoader(train_loader, batch_size=64)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    for epoch in range(10):
        net.train()
        for idx, (images, values) in enumerate(iter(train_loader)):
            images = images.to(device)
            values = values.to(device)

            # forward
            loss = loss_function(net(images), values)
            optimizer.zero_grad()

            # backward
            loss.backward()
            optimizer.step()

            # log
        if epoch % 10 == 0:
            print(loss)

    # evaluate
    valset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=torchvision.transforms.ToTensor())
    val_loader = torch.utils.data.DataLoader(valset, batch_size=1)

    def evaluate():
        net.eval()
        right, wrong = 0, 0
        for idx, (images, values) in enumerate(iter(val_loader)):
            images, values = images.to(device), values.to(device)
            with torch.no_grad():
                for pre, act in zip(net(images).argmax(dim=1), values):
                    if (pre == act).item():
                        right += 1
                    else:
                        wrong += 1
        print('accuracy:', right/(right+wrong))
    evaluate()

    # torch.save(net.state_dict(), './model')
