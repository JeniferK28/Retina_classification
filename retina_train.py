import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from train_pred import train_model
from sklearn.metrics import confusion_matrix
from ptl_cm import plot_confusion_matrix
from sklearn.model_selection import KFold
from dataloader import CustomImageDataset
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Retina')
    parser.add_argument("--seed", default=1657, help="Seed number")
    parser.add_argument("--k_folds", default=5, help="N of kfolds")
    parser.add_argument("--batch_size", default=128, help="Batch_size")
    parser.add_argument("--train_data_path", default="C:/Users/DSK/Desktop/HC/trainvalid/", help="Train data path")
    parser.add_argument("--train_labels", default="C:/Users/DSK/Desktop/HC/labels_trainvalid.txt", help="Train label path")
    parser.add_argument("--resize_size", default=256, help="Image resize size")
    parser.add_argument("--lr", default=0.0005, help="learning rate")
    parser.add_argument("--step_size", default=7, help="step size")
    parser.add_argument("--gamma", default=0.1, help="gamma")
    parser.add_argument("--num_epoch", default=50, help="Number of epochs")
    parser.add_argument("--model_path", default='C:/Users/Jen/Desktop/Retina/models', help="Model saving path")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    score_val_cm = []

    # Define data transformation
    transform_img= transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.CenterCrop(args.resize_size),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.3)
        ])

    # Divide data in kfolds
    cv = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    train_dataset = CustomImageDataset(args.train_labels,args.train_data_path, transform=transform_img)
    size= len(train_dataset)

    # Train for each split of kfolds
    for fold, (train_ids, val_ids) in enumerate(cv.split(np.arange(len(train_dataset)))):

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_subsampler)
        val_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=val_subsampler)

        # Model definition
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        # Fine-tune last layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model = model.to(device)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss().cuda(0)
        optimizer_ft = optim.Adam(model.fc.parameters(), lr=args.lr)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.step_size, gamma=args.gamma)

        # Model training
        y_val_pred, y_val_labels = train_model(train_loader, val_loader, args, model, criterion, optimizer_ft, exp_lr_scheduler, fold, num_epochs=args.num_epoch)

        # Confusion matrix calculation
        cm_val = confusion_matrix(y_val_labels.cpu(), y_val_pred.cpu())
        score_val_cm.append(cm_val)

    # Ploting confusion matrix
    class_name = np.array(['N', 'Y'])
    mean_val_cm = np.sum(np.array(score_val_cm), axis=0)
    cm_f1 = plot_confusion_matrix(mean_val_cm, class_name, normalize=True)

