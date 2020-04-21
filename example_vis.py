import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, Tensor
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from sklearn import manifold
from tqdm import tqdm

from circle_loss import convert_label_to_similarity, CircleLoss


def get_loader(is_train: bool, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset=MNIST(root="./data", train=is_train, transform=ToTensor(), download=True),
        batch_size=batch_size,
        shuffle=is_train,
    )


def plot_features(features, labels, num_classes):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    features = tsne.fit_transform(features)
    x_min, x_max = features.min(0), features.max(0)
    features = (features - x_min) / (x_max - x_min)
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(num_classes):
        plt.scatter(
            features[labels == label_idx, 0],
            features[labels == label_idx, 1],
            c=colors[label_idx],
            s=50,
            alpha=0.6
        )
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right', fontsize=10)
    plt.title('t-SNE visualization of the learned features', fontsize=30)
    plt.axis('tight')
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
    val_loader = get_loader(is_train=False, batch_size=1000)
    criterion = CircleLoss(m=0.25, gamma=80)

    for epoch in range(20):
        for img, label in tqdm(train_loader):
            model.zero_grad()
            pred = model(img)
            loss = criterion(*convert_label_to_similarity(pred, label))
            loss.backward()
            optimizer.step()

    all_features = []
    all_labels = []
    for img, label in val_loader:
        pred = model(img)
        all_features.append(pred.data.numpy())
        all_labels.append(label.data.numpy())
    all_features = np.concatenate(all_features, 0)
    all_labels = np.concatenate(all_labels, 0)
    plot_features(all_features, all_labels, 10)


if __name__ == "__main__":
    main()
