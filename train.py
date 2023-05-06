import torch
import logging
import utility_fn as uf
import os
import sys
import numpy as np

from torch.utils.tensorboard import SummaryWriter


def main(args):
    logging.info(f'python3 {" ".join(sys.argv)}')
    uf.stash_files(args.log_dir)
    logging.info(f"Initializing experiment {args.name}...")
    uf.log_arguments(args, logging.info)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, val_loader = uf.get_train_valid_dataloaders(args.data_path, args.batch_size)
    model = uf.get_model(args.model).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, min_lr=args.lr_min)
    writer = SummaryWriter(args.log_dir)

    logging.info("Initialized model, optimizer, and scheduler. Beginning training...")

    train_acc, val_acc, val_auc, history = uf.train_and_evaluate(
        model, train_loader, val_loader, optimizer, scheduler, epochs=args.epochs, device=args.device,
        log_freq=args.log_freq, log_dir=args.log_dir, writer=writer, get_history=True
    )
    torch.save(history, os.path.join(args.log_dir, 'history.pth'))
    logging.info(f"train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, val_auc: {val_auc:.4f}")

    logging.info("Loading best model and evaluating on test set...")
    uf.load_checkpoint(model, os.path.join(args.log_dir, 'best_model.pth'), args.device)
    test_loader = uf.get_test_dataloader(os.path.join(args.data_path, 'test'), args.batch_size)
    tta_transforms = uf.get_transforms('test')
    uf.infer(
        model, test_loader, tta_transforms, args.tta, args.device, args.log_freq,
        os.path.join(args.log_dir, f'submissions_{args.name}.csv')
    )


if __name__ == "__main__":
    args = uf.get_arguments()
    os.makedirs(args.log_dir, exist_ok=True)

    log_format = '%(asctime)s | %(filename)15s [ %(levelname)8s ] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S')
    fh = logging.FileHandler(os.path.join(args.log_dir, f'{args.name}.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    main(args)
