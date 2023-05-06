import torch
import logging
import os
import utility_fn as uf
import sys
import argparse
import time
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, roc_curve


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='../datasets/PCAM', help='Path to dataset')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--log-freq', type=int, default=100, help='Log interval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    args = parser.parse_args()
    return args


def main(args):
    logging.info(f'python3 {" ".join(sys.argv)}')
    uf.log_arguments(args, logging.info)
    logging.info('Calculating metrics...')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    _, val_loader = uf.get_train_valid_dataloaders(args.data_path, args.batch_size)
    experiments = os.listdir(args.log_dir)
    model_names = ['resnet34', 'mobilenet', 'densenet', 'efficientnet', 'convnext', 'inception', 'attn_inception']
    models = []
    for exp in experiments:
        for model_name in model_names:
            if exp.startswith(model_name):
                model = uf.get_model(model_name).to(args.device)
                model = uf.load_checkpoint(model, os.path.join(args.log_dir, exp, 'best_model.pth'), args.device)
                models.append(model)
                break
    
    for exp_name, model in zip(experiments, models):
        logging.info(f'Evaluating {exp_name}...')
        metrics = get_metrics(model, val_loader, args.device, args.log_freq)
        for m in metrics['classification_report'].split('\n'):
            logging.info(m)
        log_confusion_matrix(metrics['confusion_matrix'], logging.info)


@torch.no_grad()
def get_metrics(model, dataloader, device='cuda', log_freq=100):
    model.eval()
    y_true = []
    y_pred = []
    y_score = []
    for step, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_score.extend(logits[:, 1].cpu().numpy())

        if step % log_freq == 0 or step == len(dataloader) - 1:
            logging.info(
                f"Valid[{step+1:3d}/{len(dataloader):3d}] finished."
            )

    return {
        'classification_report': classification_report(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'roc_curve': roc_curve(y_true, y_score)
    }


def log_confusion_matrix(conf_matrix, log_fn=print):
    separator = '+' + '+'.join(['-'*7 for _ in range(3)]) + '+'
    log_fn(separator)
    log_fn(f'| {"T/P":^5} | {"P-0":^5} | {"P-1":^5} |')
    log_fn(separator)
    for i, row in enumerate(conf_matrix):
        log_fn(f'| {f"T-{i}":^5} | {row[0]:>5} | {row[1]:>5} |')
    log_fn(separator)


if __name__ == "__main__":
    args = get_arguments()
    os.makedirs(args.log_dir, exist_ok=True)

    log_format = '%(asctime)s | %(filename)15s [ %(levelname)8s ] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S')
    fh = logging.FileHandler(os.path.join(args.log_dir, f'metrics_{int(time.time())}.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    main(args)
