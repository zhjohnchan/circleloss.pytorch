import os
import torch
import matplotlib.pyplot as plt
from torch import nn, Tensor
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from circle_loss import convert_label_to_similarity, CircleLoss


def get_loader(is_train: bool, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset=MNIST(root="./data", train=is_train, transform=ToTensor(), download=True),
        batch_size=batch_size,
        shuffle=is_train,
    )


def plot(img_1, img_2, same):
    plt.figure(12)
    if not same:
        plt.suptitle('These two digits are different.', fontsize=20)
    else:
        plt.suptitle('These two digits are the same.', fontsize=20)
    plt.subplot(121)
    plt.imshow(img_1, cmap='Greys')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(img_2, cmap='Greys')
    plt.axis('off')
    plt.show()


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


def main() -> None:
    model = Model()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    train_loader = get_loader(is_train=True, batch_size=64)
    val_loader = get_loader(is_train=False, batch_size=2)
    criterion = CircleLoss(m=0.25, gamma=80)

    for epoch in range(20):
        for img, label in tqdm(train_loader):
            model.zero_grad()
            pred = model(img)
            loss = criterion(*convert_label_to_similarity(pred, label))
            loss.backward()
            optimizer.step()

    thresh = 0.75
    for img, label in val_loader:
        pred = model(img)
        pred_label = torch.sum(pred[0] * pred[1]) > thresh
        plot(img[0, 0].data.numpy(), img[1, 0].data.numpy(), pred_label)
        break


if __name__ == "__main__":
    main()
