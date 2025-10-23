# test.py

import argparse
import torch
from torch.utils.data import DataLoader
import os
import glob
from dataset import LensedDataset, get_val_transforms
from model import get_deit_tiny_distilled_enhanced
from evaluate import evaluate
from utils import set_seed

def main():
    parser = argparse.ArgumentParser(description='Evaluate model on .png test images')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .pth checkpoint file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    lensed_dir = os.path.join(args.data_root, 'Your data path(lensed)')
    unlensed_dir = os.path.join(args.data_root, 'Your data path(unlensed)')

    lensed_files = sorted(glob.glob(os.path.join(lensed_dir, "*.png")))
    unlensed_files = sorted(glob.glob(os.path.join(unlensed_dir, "*.png")))

    test_paths = lensed_files + unlensed_files
    test_labels = [1] * len(lensed_files) + [0] * len(unlensed_files)

    print(f"Loaded {len(test_paths)} total images: {len(lensed_files)} lensed, {len(unlensed_files)} unlensed")

    val_tf = get_val_transforms()
    test_ds = LensedDataset(test_paths, test_labels, transform=val_tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = get_deit_tiny_distilled_enhanced(num_classes=2, pretrained=False).to(device)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    print(f"Loading checkpoint from {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print("Evaluating on PNG test set...")
    evaluate(model, test_loader, device, output_dir='outputs/l-h-r')

if __name__ == '__main__':
    main()
