import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import transforms, datasets
import os
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from argparse import Namespace

from vit import ViT
from util import *

def plotTwo(l1, l2, title, savePath):
    plt.ioff()
    plt.plot(range(len(l1)), l1, color="red")
    plt.plot(range(len(l2)), l2, color ="green")
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title(title)
    plt.savefig(savePath, transparent=True)
    plt.clf()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataDir = './data/'
    plotDir = f'./plots/epoch_{args.epochs}_bs_{args.batchSize}_lr_{args.initLr}_wd_{args.weightDecay}'
    os.mkdir(plotDir)

    imageDatasets = {x: datasets.ImageFolder(os.path.join(dataDir, x), dataTransforms[x]) for x in ['train', 'val', 'test']}
    dataloaders = {
        'train': DataLoader(imageDatasets['train'], batch_size=args.batchSize, shuffle=True, num_workers=1),
        'val': DataLoader(imageDatasets['val'], batch_size=args.batchSize, shuffle=True, num_workers=1),
        'test': DataLoader(imageDatasets['test'], batch_size=args.batchSize, shuffle=False, num_workers=0),
    }
    datasetSizes = {x: len(imageDatasets[x]) for x in ['train', 'val', 'test']}
    classNames = imageDatasets['train'].classes

    model = ViT(image_size = 200, patch_size = 25, num_classes = len(classNames), dim = 1024, depth = 6, heads = 8, mlp_dim=512, dropout = 0.1, emb_dropout = 0.1).to(device)
    optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, model.parameters()), lr=args.initLr, weight_decay=args.weightDecay)
    criterion = CrossEntropyLoss()

    valLossList, valAccList = [], []
    for epoch in range(args.epochs):
        ''' Train '''
        model.train()
        trainLoss = 0.0
        correct, total = 0, 0
        lossList, accList = [], []
        for i, (x, y) in enumerate(dataloaders['train']):
            x, y = x.to(device), y.to(device)
            yHat = model(x)
            loss = criterion(yHat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # stat
            trainLoss += loss.item()
            total += yHat.size(0)
            val, idx = yHat.max(1)
            correct += idx.eq(y).sum().item()
            avgLoss = float(trainLoss)/(i+1)
            avgAcc = float(correct)/total
            lossList.append(avgLoss)
            accList.append(avgAcc)
            if(i % 100 == 0):
                print(f"Train: {i}/{len(dataloaders['train'])} - AccuLoss: {avgLoss:.3f} | AccuAcc: {avgAcc:.3f} ({correct}/{total})")
        # plot for each epoch
        plotTwo(lossList, accList, f"Epoch {epoch} Loss and Acc", os.path.join(plotDir, f"e{epoch}_train.png"))

        ''' Val '''
        model.eval()
        testLoss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloaders['val']):
                x, y = x.to(device), y.to(device)
                yHat = model(x)
                loss = criterion(yHat, y)
                
                testLoss += loss.item()
                total += yHat.size(0)
                val, idx = yHat.max(1)
                correct += idx.eq(y).sum().item()
                avgLoss = float(testLoss)/(i+1)
                avgAcc = float(correct)/total

                valLossList.append(avgLoss)
                valAccList.append(avgAcc)
                if(i % 100 == 0):
                    print(f"Val: {i}/{len(dataloaders['val'])} - AccuLoss: {avgLoss:.3f} | AccuAcc: {avgAcc:.3f} ({correct}/{total})")
        plotTwo(valLossList, valAccList, f"Validation Loss and Acc", os.path.join(plotDir, "eval.png"))
        