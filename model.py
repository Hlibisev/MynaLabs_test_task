import torch
from catalyst import dl
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18


class Reshape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
      return x.reshape(x.size(0), -1)


class CustomRunner(dl.SupervisedRunner):
    def handle_batch(self, batch):
        x, y = batch[self._input_key], batch[self._target_key]
        prediction = self.model(x)  # let's manually set teacher model_weights to eval mode

        self.batch = {
            self._input_key: x, self._output_key: prediction, self._target_key: y,
            "sogmoid_logits": torch.sigmoid(prediction), 'bin_logits': (prediction > 0.5).int()
        }

class Predict(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x > 0.5).int()


def get_model(path_to_weights=None):
    """ Model for glasses recognition """
    pretrained = True if path_to_weights is None else False
    model = torch.nn.Sequential(
        *(list(resnet18(pretrained=pretrained).children())[:5]),
        nn.AdaptiveAvgPool2d((1, 1)),
        Reshape(),
        nn.Linear(64, 1),
    )

    if path_to_weights:
        model.load_state_dict(torch.load(path_to_weights))

    return model


def get_split_lenghts(dataset):
    train_size = int(0.6 * len(dataset))
    test_size = (len(dataset) - train_size) // 2
    val_size = test_size + (len(dataset) - train_size) % 2

    return train_size, test_size, val_size


def get_transform():
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((120, 120)),
                transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    return transform


def target_transform(x):
    return torch.Tensor([x]).float()


if __name__ == "__main__":
    # training model_weights for glasses recognition
    reprocess = get_transform()

    dataset = datasets.ImageFolder('/content/data/', transform=reprocess, target_transform=target_transform)
    train_dataset, test_dataset, valid_dataset = random_split(dataset, get_split_lenghts(dataset))

    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=2, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, num_workers=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=2, shuffle=True)

    loaders = {
        "train": train_loader,
        "valid": valid_loader,
    }

    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.BCEWithLogitsLoss()

    runner = CustomRunner()

    # model_weights training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=5,
        verbose=1,
        callbacks=[
            dl.AccuracyCallback(input_key="bin_logits", target_key="targets"),
            dl.EarlyStoppingCallback(patience=2, loader_key='valid', metric_key="accuracy", minimize=False)
        ],
        valid_loader="valid",
        minimize_valid_metric=False,
    )

    print(runner.evaluate_loader(
        model=nn.Sequential(model, Predict()),
        loader=test_loader,
        callbacks=[
            dl.AccuracyCallback(input_key="logits", target_key="targets")
        ]
        )['accuracy']
    )
