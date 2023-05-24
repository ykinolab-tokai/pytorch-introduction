import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as audio_transforms

from introduction.model import SimpleDNN
from introduction.dataset import AcousticSceneDataset
from introduction.trainer import ClassifierTrainer
import introduction.transform as transform



def get_data_loaders():
    transforms = transform.Zip(
        transform.Compose([
            audio_transforms.Resample(44100, 8000),
            audio_transforms.Spectrogram(n_fft=512),
            audio_transforms.AmplitudeToDB()
        ]),
        transform.Identity()
    )

    testset = AcousticSceneDataset(
        root="../data/small-acoustic-scenes",
        mode="test",
        transforms=transforms
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    return testloader, testset.LABELS


def examples(loader, net, classes):
    dataiter = iter(loader)
    waveforms, labels = next(dataiter)
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[i]] for i in range(len(labels))))

    outputs = net(waveforms)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[i]] for i in range(len(labels))))


def calculate_accuracy(loader, net, classes):
    num_classes = len(classes)
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    with torch.no_grad():
        for data in loader:
            waveforms, labels = data
            outputs = net(waveforms)
            _, predicted = torch.max(outputs, 1)
            # c = (predicted == labels).squeeze()
            c = (predicted == labels)
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(num_classes):
        print('Accuracy of %5s : %2d %%' % (
              classes[i], 100 * class_correct[i] / class_total[i]))
    
    print('Total accuracy : %2d %%' % (
          100 * sum(class_correct) / sum(class_total)))



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader, classes = get_data_loaders()

    net = SimpleDNN()
    net.load_state_dict(torch.load("../model/trained_weights.pth"))
    net.eval()

    examples(loader, net, classes)
    calculate_accuracy(loader, net, classes)
