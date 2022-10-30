from argparse import Namespace
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

argsDict = {
    "epochs": 80,
    "batchSize": 32,
    "initLr": 1e-5,
    "weightDecay": 1e-3,
}
args = Namespace(**argsDict)


dataTransforms = {
    'train': transforms.Compose([
        transforms.CenterCrop(200),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),  # type: ignore
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet norm
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(200),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.CenterCrop(200),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def plotTwo(l1, l2, title, savePath):
    plt.ioff()
    plt.plot(range(len(l1)), l1, color="red")
    plt.plot(range(len(l2)), l2, color ="green")
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title(title)
    plt.savefig(savePath)
    plt.clf()