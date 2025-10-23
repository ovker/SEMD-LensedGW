# train.py

import os
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def train(model, dataloaders, device,
          num_epochs=10, learning_rate=1e-4,
          checkpoint_dir='checkpoints', model_name='best_model'):

    os.makedirs(checkpoint_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)


    best_val_auc = 0.0
    best_model_path = None

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        # ------ Training Phase ------
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for images, labels in dataloaders['train']:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, dim=1)
            running_correct += (preds == labels).sum().item()
            total += batch_size

        epoch_loss = running_loss / total
        epoch_acc = running_correct / total
        print(f"  Train loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}")

        # ------ Validation Phase ------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for images, labels in dataloaders['val']:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                batch_size = images.size(0)
                val_loss += loss.item() * batch_size
                _, preds = torch.max(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += batch_size

                probs = nn.functional.softmax(outputs, dim=1)[:, 1]
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(probs.cpu().numpy())

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(all_labels, all_scores)
        print(f"  Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

        # ------ Save Best Model ------
        # Always save to the same filename: checkpoint_dir/model_name.pth
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_path = os.path.join(checkpoint_dir, f"{model_name}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  [Saved Best] {best_model_path}")

    print(f"\nTraining complete. Best val AUC: {best_val_auc:.4f}")
    return best_model_path
