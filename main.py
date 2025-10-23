import argparse
import torch
from torch.utils.data import DataLoader
import os

from dataset import load_data, LensedDataset, get_train_transforms, get_val_transforms
from model import get_deit_tiny_distilled_enhanced
from train import train
from evaluate import evaluate
from utils import set_seed


def main():
    parser = argparse.ArgumentParser(description='Train GW Image Classifier')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory of data')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name for saving the best model')
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dirs = {
        'lensed': os.path.join(args.data_root, 'Your data path(lensed)'),
        'unlensed': os.path.join(args.data_root, 'Your data path(unlensed)')
    }

    splits = load_data(data_dirs, split_ratios=(0.75, 0.125, 0.125), random_seed=42)
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    dataloaders = {}
    for phase in ['train', 'val', 'test']:
        paths, labels = splits[phase]
        if phase == 'train':
            dataset = LensedDataset(paths, labels, transform=train_transform)
            shuffle = True
        else:
            dataset = LensedDataset(paths, labels, transform=val_transform)
            shuffle = False

        dataloaders[phase] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )

    model = get_deit_tiny_distilled_enhanced(
        num_classes=2,
        pretrained=True,
        hidden_dim=512,
        dropout_rate=0.5,
        freeze_backbone=False
    )
    model = model.to(device)

    best_model_path = train(
        model,
        {'train': dataloaders['train'], 'val': dataloaders['val']},
        device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir='checkpoints',
        model_name=args.model_name
    )
    print(f"\nBest model saved at: {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print("Loaded best model for evaluation. Evaluating on test set...")
    evaluate(model, dataloaders['test'], device, output_dir='output')


if __name__ == '__main__':
    main()
