import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import argparse
import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchsummary import summary

from tqdm import tqdm
import medmnist
from medmnist import INFO, Evaluator
from CustomDataset.datasets import BoneTumorDataset
import torchattacks
from torchattacks import PGD, FGSM
from MedViT import MedViT_small, MedViT_base, MedViT_large
import wandb
from torchvision.transforms.transforms import Resize
from torchvision.transforms import InterpolationMode


parser = argparse.ArgumentParser(description='BTXRD Classification')
parser.add_argument('--model_path', type=str, default='exp', metavar='N',
                    help='Name of the experiment')
parser.add_argument('--pretrain_path', type=str, default='pretrain/yolov8n-cls.pt', metavar='N',
                    help='Name of the experiment')
parser.add_argument('--path_yolo_yaml', type=str, default='yolo/cfg/models/11/yolo11-cls-lka.yaml', metavar='N',
                    help='Name of the experiment')
parser.add_argument('--model_name', type=str, default='convnext', metavar='N',
                    help='Name of the model')
parser.add_argument('--yolo_scale', default='n', choices=['n','s','m','l','x'])
parser.add_argument('--img_size', type=int, default=608, metavar='img_size',
                    help='Size of input image)')
parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of episode to train')
parser.add_argument('--use_sgd', action='store_true', default=False, help='Use SGD')
parser.add_argument('--scenario', default='A', type=str,help='A=no clahe, B=clahe as weak aug, C=clahe as preprocessing')
parser.add_argument('--use_balanced_weight', action='store_true', default=False, help='Use Weight Balancing')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--dropout', type=float, default=0.2, metavar='LR',
                    help='Dropout')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num_worker', type=int, default=4, metavar='S',
                    help='Num of Worker')
parser.add_argument('--eval', type=bool,  default=False,
                    help='evaluate the model')
parser.add_argument('--project_name', type=str, default='BTXRD', metavar='N',
                    help='Name of the Project WANDB')

args = parser.parse_args()


data_flag = 'btxrd'
# [tissuemnist, pathmnist, chestmnist, dermamnist, octmnist,
# pnemoniamnist, retinamnist, breastmnist, bloodmnist, tissuemnist, organamnist, organcmnist, organsmnist]
# download = True

NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
lr = 0.005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# info = INFO[data_flag]
# task = info['task']
# n_channels = info['n_channels']
# n_classes = len(info['label'])

n_classes = 3
# DataClass = getattr(medmnist, info['python_class'])

# print("number of channels : ", n_channels)
# print("number of classes : ", n_classes)

# preprocessing
train_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size), interpolation=InterpolationMode.BILINEAR),
    transforms.Lambda(lambda image: image.convert('RGB')),
    torchvision.transforms.AugMix(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])
test_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size), interpolation=InterpolationMode.BILINEAR),
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

DATASET_DIR = '../btxrd/data/BTXRD'
metadata_xlsx_path = os.path.join(DATASET_DIR, 'dataset.xlsx')
train_path = os.path.join(DATASET_DIR, 'train.xlsx')
test_path = os.path.join(DATASET_DIR, 'val.xlsx')  
IMG_DIR = os.path.join(DATASET_DIR, 'images')

# load the data
train_dataset = BoneTumorDataset(
    split_xlsx_path=train_path,
    metadata_xlsx_path=metadata_xlsx_path,
    image_dir=IMG_DIR,  # make sure this exists
    transform=train_transform
)

test_dataset = BoneTumorDataset(
    split_xlsx_path=test_path,
    metadata_xlsx_path=metadata_xlsx_path,
    image_dir=IMG_DIR,
    transform=test_transform
)

# pil_dataset = DataClass(split='train', download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)



model = MedViT_small(num_classes = n_classes).to(device)

model.load_state_dict(torch.load(args.model_path, weights_only=True))
#model = MedViT_base(num_classes = n_classes).cuda()
#model = MedViT_large(num_classes = n_classes).cuda()


task = ''
# define loss function and optimizer
if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()


split = 'test'

model.eval()
y_true = torch.tensor([])
y_score = torch.tensor([])

data_loader = train_loader_at_eval if split == 'train' else test_loader

with torch.no_grad():
    for inputs, targets in tqdm(data_loader):
        inputs = inputs.cuda()
        outputs = model(inputs)
        outputs = outputs.softmax(dim=-1)
        y_score = torch.cat((y_score, outputs.cpu()), 0)

    y_score = y_score.detach().numpy()

    evaluator = Evaluator(data_flag, split, size=224)
    metrics = evaluator.evaluate(y_score)

    print('%s  auc: %.3f  acc: %.3f' % (split, *metrics))

BATCH_SIZE = 5
test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)    
model.eval()

correct = 0
total = 0

atk = FGSM(model, eps=0.01)

for images, labels in tqdm(test_loader):
    labels = labels.squeeze(1)
    images = atk(images, labels).cuda()
    outputs = model(images)

    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('FGSM Robust accuracy: %.2f %%' % (100 * float(correct) / total))


model.eval()

correct = 0
total = 0

atk = PGD(model, eps=8/255, alpha=4/255, steps=10, random_start=True)

for images, labels in test_loader:
    labels = labels.squeeze(1)
    images = atk(images, labels).cuda()
    outputs = model(images)

    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('PGD Robust accuracy: %.2f %%' % (100 * float(correct) / total))