import torch
import torch.nn as nn
import numpy as np
import logging
import os
import glob
import shutil
import argparse
import time
import pandas as pd
import cv2
import torch.nn.functional as F
from networks import AttentionInceptionV3, AttentionResNet34, AttentionMobileNetV3
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from matplotlib import patches
from sklearn.metrics import roc_auc_score
from PIL import Image


def accuracy(preds, labels):
    """
    Compute accuracy of predictions for classification task.

    @param preds: (torch.Tensor) logits of shape (batch_size, num_classes)
    @param labels: (torch.Tensor) tensor of shape (batch_size,)
    @return: (float) accuracy
    """
    preds = torch.argmax(preds, dim=1)
    return torch.sum(preds == labels).item() / len(labels)


def get_arguments():
    """
    Get parser for command line arguments.

    @return: (argparse.Namespace) arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='Name of experiment')
    parser.add_argument('--model', type=str, default='resnet34', help='Model to use')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train')
    parser.add_argument('--tta', type=int, default=5, help='Number of TTA augmentations')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--lr-min', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--log-freq', type=int, default=100, help='Log interval')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--data-path', type=str, default='../datasets/PCAM', help='Path to dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir, f'{args.name}_{int(time.time())}')
    args.device = torch.device(args.device)

    return args


def stash_files(log_dir):
    """
    Stash all python files in log_dir.
    
    @param log_dir: (str) log directory
    """
    py_files = glob.glob('*.py')
    stash_dir = os.path.join(log_dir, 'stash')
    os.makedirs(stash_dir, exist_ok=True)
    for py_file in py_files:
        shutil.copy(py_file, stash_dir)


def load_checkpoint(model, checkpoint_path, map_location='cuda'):
    """
    Load model weights from checkpoint.

    @param model: (torch.nn.Module) model
    @param checkpoint_path: (str) path to checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def save_checkpoint(model, checkpoint_path):
    """
    Save model weights to checkpoint.

    @param model: (torch.nn.Module) model
    @param checkpoint_path: (str) path to checkpoint
    """
    torch.save({
        'model_state_dict': model.state_dict(),
    }, checkpoint_path)


def log_arguments(args, log_fn=print):
    """
    Log arguments.

    @param args: (argparse.Namespace) arguments
    @param log_fn: (function) logging function
    """
    separator = "+" + "-" * 27 + "+" + "-" * 32 + "+"

    log_fn(separator)
    log_fn(f'| {"Argument":^25} | {"Value":^30} |')
    log_fn(separator)
    for arg in vars(args):
        log_fn(f'| {str(arg).upper():<25} | {str(getattr(args, arg)):>30} |')
    log_fn(separator)


def get_transforms(split):
    """
    Get transforms for training and validation.

    @param split: (str) split to get transforms for (train/valid/test)
    @return:
        transform: (torchvision.transforms) transform
    """
    if split == 'train':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.GaussianBlur(3),
            transforms.ToTensor(),
        ])
    elif split == 'valid':
        transform = transforms.ToTensor()
    elif split == 'test':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.GaussianBlur(3),
        ])
    else:
        raise ValueError(f'Invalid split: {split}')
    
    return transform


def get_train_valid_dataloaders(path='../datasets/PCAM', batch_size=32, train_transform=None, valid_transform=None):
    """
    Get train, validation data loaders.

    @param batch_size: (int) batch size
    @param train_transform: (torchvision.transforms) transform for training data
    @param valid_transform: (torchvision.transforms) transform for test data
    @return:
        train_loader: (torch.utils.data.DataLoader) train dataloader
        val_loader: (torch.utils.data.DataLoader) validation dataloader
    """
    if train_transform is None:
        train_transform = get_transforms('train')
    train_data = datasets.ImageFolder(os.path.join(path, 'train'), transform=train_transform)

    if valid_transform is None:
        valid_transform = get_transforms('valid')
    valid_data = datasets.ImageFolder(os.path.join(path, 'valid'), transform=valid_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_full_train_valid_dataloaders(path='../datasets/PCAM', batch_size=32, train_transform=None, valid_transform=None):
    """
    Get full train-set, validation data loaders.
    Contains 1.5x more train data than get_train_valid_dataloaders.

    @param batch_size: (int) batch size
    @param train_transform: (torchvision.transforms) transform for training data
    @param valid_transform: (torchvision.transforms) transform for test data
    @return:
        train_loader: (torch.utils.data.DataLoader) train dataloader
        val_loader: (torch.utils.data.DataLoader) validation dataloader
    """
    if train_transform is None:
        train_transform = get_transforms('train')
    train_data = datasets.PCAM(root=path, split='train', transform=train_transform)

    if valid_transform is None:
        valid_transform = get_transforms('valid')
    valid_data = datasets.ImageFolder(os.path.join(path, 'valid'), transform=valid_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_test_dataloader(path='../datasets/PCAM/test', batch_size=32):
    """
    Get test data loader.

    @param batch_size: (int) batch size
    @return:
        test_loader: (torch.utils.data.DataLoader) test dataloader
    """
    test_data = TifDataset(path, transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return test_loader


def get_model(model_name, num_classes=2, get_attn=False):
    """
    Get model.

    @param model_name: (str) model name
    @param num_classes: (int) number of classes
    @param get_attn: (bool) whether to get attention weights for attention models
    """
    if model_name == 'resnet34':
        model = models.resnet34(num_classes=num_classes)
    elif model_name == 'mobilenet':
        model = models.mobilenet_v3_large(num_classes=num_classes)
    elif model_name == 'densenet':
        model = models.densenet121(num_classes=num_classes)
    elif model_name == 'efficientnet':
        model = models.efficientnet_v2_l(num_classes=num_classes)
    elif model_name == 'convnext':
        model = models.convnext_small(num_classes=num_classes)
    elif model_name == 'inception':
        model = models.inception_v3(num_classes=num_classes, aux_logits=False, init_weights=False)
    elif model_name == 'attn_inception':
        model = AttentionInceptionV3(num_classes=num_classes, get_attn=get_attn)
    elif model_name == 'attn_resnet34':
        model = AttentionResNet34(num_classes=num_classes, get_attn=get_attn)
    elif model_name == 'attn_mobilenet':
        model = AttentionMobileNetV3(num_classes=num_classes, get_attn=get_attn)
    else:
        raise ValueError(f'Invalid model name: {model_name}.')
    return model


def plot_images(images, true_labels=None, pred_labels=None, attn=None, label_names=['benign', 'malignant']):
    """
    Plot images with labels, predictions and attention maps.
    Incorrectly classified images are highlighted in red.

    @param images: (torch.Tensor) images (B, C, H, W)
    @param true_labels: (torch.Tensor) true labels (B)
    @param pred_labels: (torch.Tensor) predicted labels (B)
    @param attn: (torch.Tensor) attention maps (B, 1, H', W')
    @param label_names: (list) label names
    @return:
        ax: (matplotlib.axes.Axes) axes
    """
    nrow, H, W = int(np.sqrt(len(images))), images.shape[-2], images.shape[-1]
    grid = make_grid(images, nrow, normalize=True, scale_each=True)
    grid = grid.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(2 * nrow, 2 * nrow))
    ax.axis('off')

    if attn is not None:
        attn = F.interpolate(attn, size=(H, W), mode='bilinear', align_corners=True)
        attn = make_grid(attn, nrow, normalize=True, scale_each=True)
        attn = attn.permute(1, 2, 0).mul(255).byte().cpu().numpy()
        attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
        attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
        attn = np.float32(attn) / 255
        grid = 0.6 * grid + 0.4 * attn
    ax.imshow(grid)

    if true_labels is None: return
    true_labels = [label_names[label.item()] for label in true_labels]
    for i, label in enumerate(true_labels):
        # The black borders are 2px wide. We add a 3px margin to the text.
        x_pos = (i % nrow) * W + (i % nrow + 1) * 2 + 3
        y_pos = (i // nrow) * H + (i // nrow + 1) * 2 + 3
        ax.text(x_pos, y_pos, label, va='top', color='white', bbox=dict(facecolor='black', alpha=0.5))
    
    if pred_labels is None: return
    pred_labels = [label_names[label.item()] for label in pred_labels]
    for i, label in enumerate(pred_labels):
        if label == true_labels[i]: continue
        x_pos = (i % nrow) * (W + 2)
        y_pos = (i // nrow) * (H + 2)
        rec = patches.Rectangle((x_pos, y_pos), W+2, H+2, linewidth=2.5, edgecolor='r', facecolor='none')
        ax.add_patch(rec)

    plt.show()

    return ax


def train_and_evaluate(
        model, train_loader, valid_loader, optimizer, scheduler, epochs=10, device='cuda', log_freq=100, 
        log_dir='logs', writer=None, get_history=False
    ):
    """
    Train and evaluate the model.

    @param model: (torch.nn.Module) model to train
    @param train_loader: (torch.utils.data.DataLoader) train dataloader
    @param valid_loader: (torch.utils.data.DataLoader) validation dataloader
    @param optimizer: (torch.optim) optimizer
    @param scheduler: (torch.optim.lr_scheduler) scheduler
    @param epochs: (int) number of epochs
    @param device: (torch.device) device to use
    @param log: (int) log interval
    @param log_dir: (str) log directory
    @param writer: (torch.utils.tensorboard.SummaryWriter) tensorboard writer
    @param get_history: (bool) whether to return history
    @return: (float, float, float, dict) best train accuracy, best validation accuracy, best AUC, history
    """
    criterion = nn.CrossEntropyLoss()
    global_train_acc, global_train_loss = [], []
    global_valid_acc, global_valid_loss, global_valid_auc = [], [], []
    best_train_acc, best_valid_acc, best_valid_auc = 0, 0, 0

    for epoch in range(epochs):
        train_acc, train_loss = train(model, train_loader, criterion, optimizer, scheduler, epoch, device, log_freq)
        global_train_acc.append(train_acc)
        global_train_loss.append(train_loss)

        valid_acc, valid_loss, valid_auc = evaluate(model, valid_loader, criterion, epoch, device, log_freq)
        global_valid_acc.append(valid_acc)
        global_valid_loss.append(valid_loss)
        global_valid_auc.append(valid_auc)

        best_train_acc = max(best_train_acc, train_acc)
        best_valid_auc = max(best_valid_auc, valid_auc)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            save_checkpoint(model, os.path.join(log_dir, 'best_model.pth'))

        logging.info(
            f"[{epoch+1:2d}/{epochs}] " + 
            f"best@train-acc: {best_train_acc:.2%}, " +
            f"best@val-acc: {best_valid_acc:.2%}, " + 
            f"best@val-auc: {best_valid_auc:.4f}"
        )
        if writer is not None:
            writer.add_scalar('Train/Acc', train_acc, epoch)
            writer.add_scalar('Train/Loss', train_loss, epoch)
            writer.add_scalar('Valid/Acc', valid_acc, epoch)
            writer.add_scalar('Valid/Loss', valid_loss, epoch)
            writer.add_scalar('Valid/AUC', valid_auc, epoch)

    history = {
        'train_acc': global_train_acc,
        'train_loss': global_train_loss,
        'valid_acc': global_valid_acc,
        'valid_loss': global_valid_loss,
        'valid_auc': global_valid_auc
    }
    if get_history:
        return best_train_acc, best_valid_acc, best_valid_auc, history
    else:
        return best_train_acc, best_valid_acc, best_valid_auc


def train(model, train_loader, criterion, optimizer, scheduler, epoch, device='cuda', log_freq=100):
    """
    Train the model.

    @param model: (torch.nn.Module) model to train
    @param train_loader: (torch.utils.data.DataLoader) train dataloader
    @param criterion: (torch.nn) loss function
    @param optimizer: (torch.optim) optimizer
    @param scheduler: (torch.optim.lr_scheduler) scheduler
    @param epoch: (int) current epoch
    @param device: (torch.device) device to use
    @param log: (int) log interval
    @return: (float, float) train loss, train accuracy
    """
    model.train()

    train_acc, train_loss = [], []
    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)

        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        acc = accuracy(logits, labels)

        train_loss.append(loss.item())
        train_acc.append(acc)

        if step % log_freq == 0 or step == len(train_loader) - 1:
            logging.info(
                f"Epoch[{epoch+1:2d} - {step+1:4d}/{len(train_loader):4d}] " +
                f"Train-Loss: {np.mean(train_loss):.4f} Train-Acc: {np.mean(train_acc):.2%}"
            )
    train_acc, train_loss = np.mean(train_acc), np.mean(train_loss)
    scheduler.step(train_loss)

    return train_acc, train_loss


@torch.no_grad()
def evaluate(model, valid_loader, criterion, epoch, device='cuda', log_freq=100):
    """
    Evaluate the model.

    @param model: (torch.nn.Module) model to evaluate
    @param test_loader: (torch.utils.data.DataLoader) test dataloader
    @param device: (torch.device) device to use
    @param log_freq: (int) log interval
    @return: (float, float, float) valid loss, valid accuracy, roc_auc_score
    """
    model.eval()
    
    valid_loss, valid_acc = [], []
    y_true, y_pred = [], []
    for step, (images, labels) in enumerate(valid_loader):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        
        loss = criterion(logits, labels)
        acc = accuracy(logits, labels)
        
        valid_loss.append(loss.item())
        valid_acc.append(acc)
        y_true.append(labels.cpu().numpy())
        y_pred.append(logits.cpu().numpy())

        if step % log_freq == 0 or step == len(valid_loader) - 1:
            logging.info(
                f"Epoch[{epoch+1:2d} - {step+1:4d}/{len(valid_loader):4d}] " +
                f"Valid-Loss: {np.mean(valid_loss):.4f} Valid-Acc: {np.mean(valid_acc):.2%}"
            )
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    auc = roc_auc_score(y_true, y_pred[:, 1])

    return np.mean(valid_acc), np.mean(valid_loss), auc


def tta(model, x, transform, n=5):
    """
    Test time augmentation.

    @param model: (torch.nn.Module) model
    @param x: (torch.Tensor) input image (channel, height, width)
    @param transform: (torchvision.transforms) transform
    @param n: (int) number of augmentations
    @return:
        tta_logits: (torch.Tensor) logits (num_classes,)
    """
    tta_logits = []
    for _ in range(n):
        x_tta = transform(x)
        logits = model(x_tta)
        tta_logits.append(logits)
    return torch.mean(torch.stack(tta_logits), dim=0)


@torch.no_grad()
def infer(
        model, test_loader, tta_transform, tta_n=5, device='cuda', log_freq=100,
        csv_path='submissions.csv'
    ):
    """
    Run inference on the model.

    @param model: (torch.nn.Module) model to evaluate
    @param test_loader: (torch.utils.data.DataLoader) test dataloader
    @param tta_transform: (torchvision.transforms) test time augmentation transforms
    @param device: (torch.device) device to use
    @param log_freq: (int) log interval
    """
    model.eval()

    results = {'id': [], 'label': []}
    for step, (img_id, images) in enumerate(test_loader):
        images = images.to(device)
        logits = tta(model, images, tta_transform, tta_n)
        labels = torch.argmax(logits, dim=1)

        results['id'].extend(img_id)
        results['label'].extend(labels.cpu().numpy())
    
        if step % log_freq == 0 or step == len(test_loader) - 1:
            logging.info(
                f"Test[{step+1:4d}/{len(test_loader):4d}] finished."
            )
    
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    logging.info(f"Submission saved to {csv_path}.")


class TifDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = glob.glob(os.path.join(root_dir, '*.tif'))
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        img_id = os.path.basename(self.imgs[idx]).split('.')[0]
        if self.transform:
            img = self.transform(img)
        return img_id, img
