import torch
from torch import nn, Tensor
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from circle_loss import convert_label_to_similarity, CircleLoss


def get_loader(is_train: bool, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset=MNIST(root="./data", train=is_train, transform=ToTensor(), download=True),
        batch_size=batch_size,
        shuffle=is_train,
    )


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )

    def forward(self, inp: Tensor) -> Tensor:
        feature = self.feature_extractor(inp).mean(dim=[2, 3])
        return nn.functional.normalize(feature)


class Classifier(nn.Module):
    def __init__(self) -> None:
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(32, 10)

    def forward(self, inp: Tensor) -> Tensor:
        return self.classifier(inp)


def main() -> None:
    model = Model()
    classifier = Classifier()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    optimizer_cls = SGD(classifier.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    train_loader = get_loader(is_train=True, batch_size=64)
    val_loader = get_loader(is_train=False, batch_size=1000)
    criterion = CircleLoss(m=0.25, gamma=80)
    criterion_xe = nn.CrossEntropyLoss()

    for epoch in range(20):
        for img, label in train_loader:
            model.zero_grad()
            features = model(img)
            loss = criterion(*convert_label_to_similarity(features, label))
            loss.backward()
            optimizer.step()
        print('[{}/{}] Training with Circle Loss.'.format(epoch + 1, 20))

    for epoch in range(20):
        for img, label in train_loader:
            model.zero_grad()
            classifier.zero_grad()
            features = model(img)
            output = classifier(features)
            loss = criterion_xe(output, label)
            loss.backward()
            optimizer_cls.step()
        print('[{}/{}] Training classifier.'.format(epoch + 1, 20))

        correct = 0
        for img, label in val_loader:
            features = model(img)
            output = classifier(features)
            pred = output.data.max(1)[1]
            correct += pred.eq(label.data).cpu().sum()
        print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
            correct, len(val_loader.dataset), 100. * correct / len(val_loader.dataset)))


if __name__ == "__main__":
    main()
