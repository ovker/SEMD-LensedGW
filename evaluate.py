
# evaluate.py
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
def evaluate(model, dataloader, device, output_dir='outputs'):

    os.makedirs(output_dir, exist_ok=True)
    model.eval()


    try:
        all_paths = dataloader.dataset.image_paths
    except:
        raise AttributeError("Dataset must expose a .image_paths attribute.")

    all_labels, all_preds, all_scores = [], [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            scores  = probs[:, 1]
            preds   = (scores >= 0.5).long()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_scores = np.array(all_scores)

    accuracy = (all_preds == all_labels).mean()
    print(f"Test Accuracy: {accuracy:.4f}")

    lensed_txt   = os.path.join(output_dir, 'lensed_results.txt')
    unlensed_txt = os.path.join(output_dir, 'unlensed_results.txt')
    with open(lensed_txt,   'w') as f_lens, \
         open(unlensed_txt, 'w') as f_unlens:
        for path, true_label, pred in zip(all_paths, all_labels, all_preds):
            fname = os.path.basename(path)
            line  = f"{fname}\t{int(pred)}\n"
            if true_label == 1:
                f_lens.write(line)
            else:
                f_unlens.write(line)
    print(f"Lensed predictions saved to   {lensed_txt}")
    print(f"Unlensed predictions saved to {unlensed_txt}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Unlensed', 'Lensed'],
                yticklabels=['Unlensed', 'Lensed'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    plt.close()

    fpr, tpr, _ = roc_curve(all_labels, all_scores, drop_intermediate=False)

    fpr_unique, unique_indices = np.unique(fpr, return_index=True)
    tpr_unique = tpr[unique_indices]

    fpr_log_grid = np.logspace(np.log10(1e-4), np.log10(1.0), 400)
    fpr_combined = np.unique(np.concatenate([fpr_log_grid, fpr_unique]))

    interp_fn = PchipInterpolator(fpr_unique, tpr_unique)
    tpr_smooth = interp_fn(fpr_combined)

    roc_data_path = os.path.join(output_dir, 'roc_data.txt')
    np.savetxt(roc_data_path,
               np.vstack([fpr_combined, tpr_smooth]).T,
               header='fpr\ttpr',
               fmt='%.6e', delimiter='\t')
    print(f"Smoothed ROC data saved to {roc_data_path}")

    plt.figure()
    plt.plot(fpr_combined, tpr_smooth, color='darkorange', lw=2, label='Smoothed ROC')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([1e-4, 1])
    plt.ylim([1e-2, 1])
    plt.xlabel('False Positive Rate (log scale)')
    plt.ylabel('True Positive Rate (log scale)')
    plt.title('ROC Curve (Zoom on low FPR)')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.legend()
    zoom_path = os.path.join(output_dir, 'roc_curve_zoom.png')
    plt.savefig(zoom_path)
    print(f"Zoomed ROC curve saved to {zoom_path}")
    plt.close()



