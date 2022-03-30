import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from neural_network import NeuralNetwork
import torch.nn.functional as F


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
model = NeuralNetwork().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.1)



def train_loop(dataloader, model:NeuralNetwork, optimizer):
    size = len(dataloader.dataset)
    print(f'LR: {scheduler.get_lr()}')
    for batch, (imgs, classes) in enumerate(dataloader):
        imgs = imgs.cuda()
        classes = classes.cuda()
        # Compute prediction and loss
        classes = F.one_hot(classes,len(labels_map)).float()
        pred = model(imgs)
        loss = model.calc_loss(pred, classes)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(imgs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    scheduler.step()


def test_loop(dataloader, model:NeuralNetwork):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    correct =  0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.cuda()
            y = y.cuda()
            pred = model(X)
            pred = model.parse_predictions(pred)
            correct += (pred == y).type(torch.float).sum().item()

    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%\n")

for i in range(50):
    train_loop(train_dataloader,model,optimizer)
    if i%4 == 0 and i > 0:
        scheduler.step()
    test_loop(test_dataloader,model)