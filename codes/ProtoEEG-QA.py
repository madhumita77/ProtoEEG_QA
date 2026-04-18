import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import time
import cv2
import gc
import torch
import random
import torch.nn as nn
import tensorflow as tf
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from sklearn.model_selection import train_test_split 
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset
from torchvision import models, transforms
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, classification_report, auc,
    confusion_matrix, cohen_kappa_score, matthews_corrcoef,
    balanced_accuracy_score, top_k_accuracy_score, brier_score_loss
)
from scipy.stats import sem


train_csv = "train_80.csv"
val_csv = "val_10.csv"
test_csv = "test_10.csv"
SPECTROGRAM_DIR = "train_spectrograms"
BASE_DIR = "hms-harmful-brain-activity-classification/"
PREPROCESSED_DIR = "preprocessed/"
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

print(f"Number of rows in train set: {len(train_df)}")
print(f"Number of rows in validation set: {len(val_df)}")
print(f"Number of rows in test set: {len(test_df)}")

brain_activities = ['Seizure', 'GPD', 'LRDA', 'Other', 'GRDA', 'LPD']
activity_mapping = {activity: idx for idx, activity in enumerate(brain_activities)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# Pre-processing Data
# -----------------------------
class OptimizedBrainDataset(Dataset):
    def __init__(self, csv_file, base_dir, activity_mapping, preprocessed_dir="preprocessed"):
        self.df = pd.read_csv(csv_file)
        self.base_dir = base_dir
        self.activity_mapping = activity_mapping
        self.preprocessed_dir = preprocessed_dir
        self.resize_transform = transforms.Resize((224, 224))
        
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        
        # Memory map preprocessed files
        self.spect_mmaps = {}
        self.spect_paths = {}
        spect_ids = self.df["spectrogram_id"].unique()
        for spect_id in spect_ids:
            npy_path = f"{self.preprocessed_dir}/{spect_id}.npy"
            if not os.path.exists(npy_path):
                self._preprocess_and_save(spect_id)
            self.spect_mmaps[spect_id] = np.load(npy_path, mmap_mode='r')
            # self.spect_paths[spect_id] = npy_path

    def __len__(self):
        return len(self.df)

    def _preprocess_and_save(self, spect_id):
        """Batch process and save spectrogram once"""
        parquet_path = f'{self.base_dir}/train_spectrograms/{spect_id}.parquet'
        temp_df = pd.read_parquet(parquet_path).drop('time', axis=1)
        
        # Process entire spectrogram
        arr = temp_df.to_numpy()
        arr = np.log1p(arr)
        arr /= arr.max()
        arr = np.nan_to_num(arr, nan=1e-4)
        arr_uint8 = (255 * arr).astype(np.uint8)
        
        np.save(f"{self.preprocessed_dir}/{spect_id}.npy", arr_uint8)

    def __getitem__(self, idx):
        spect_id, label, offset = self.df.iloc[idx][["spectrogram_id", "expert_consensus", "spectrogram_label_offset_seconds"]]
        start = int(offset) // 2
        
        # Direct memory access
        spectrogram = self.spect_mmaps[spect_id]
        # spectrogram = np.load(self.spect_paths[spect_id], mmap_mode='r')
        segment = spectrogram[start:start+300]
        
        # Convert to RGB
        rgb_image = cv2.applyColorMap(segment, cv2.COLORMAP_JET)
        rgb_image = rgb_image.astype(np.float32) / 255.0
        tensor_image = torch.tensor(rgb_image).permute(2, 0, 1)
        return self.resize_transform(tensor_image), torch.tensor(self.activity_mapping[label])

# -----------------------------
# Processing Dataset
# -----------------------------
print("\n========================")
print("\n=== Data Preparation ===")
print("\n========================")
print("Data Processing...")
train_dataset = OptimizedBrainDataset(train_csv, BASE_DIR, activity_mapping)
test_dataset = OptimizedBrainDataset(test_csv, BASE_DIR, activity_mapping)
val_dataset = OptimizedBrainDataset(val_csv, BASE_DIR, activity_mapping)
print("Data Processing done")

# -----------------------------
# Attention Module
# -----------------------------
class AttentionModule(nn.Module):
    def __init__(self, input_dim=2560):  # 1280 (support) + 1280 (query)
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, z_concat):
        x = F.relu(self.fc1(z_concat))
        x = F.relu(self.fc2(x))
        attn = F.softplus(self.fc3(x)).squeeze(-1)  # [N*k]
        return attn

# -----------------------------
# Attention-Based Prototypical Network
# -----------------------------
class AttentionPrototypicalNetwork(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.attention = AttentionModule(input_dim=2560)

    def compute_attention_weighted_prototypes(self, z_support, sy, z_query):
        prototypes = []
        for cls in torch.unique(sy):
            # Get support features of current class
            z_cls = z_support[sy == cls]  # [k, 1280]
            k = z_cls.size(0)

            # Repeat query vector k times and concatenate
            z_q_repeat = z_query.unsqueeze(0).expand(k, -1)  # [k, 1280]
            z_concat = torch.cat([z_cls, z_q_repeat], dim=1)  # [k, 2560]

            # Compute attention scores and normalize
            attn_scores = self.attention(z_concat) + 1e-8
            attn_weights = attn_scores / attn_scores.sum()

            # Weighted sum
            prototype = (attn_weights.unsqueeze(1) * z_cls).sum(0)  # [1280]
            prototypes.append(prototype)

        return torch.stack(prototypes)  # [N, 1280]

    def forward(self, sX, sy, qX):
        z_support = F.normalize(self.backbone(sX), p=2, dim=1)  # [N*k, 1280]
        z_query = F.normalize(self.backbone(qX), p=2, dim=1)    # [num_queries, 1280]

        logits = []
        for z_q in z_query:
            prototypes = self.compute_attention_weighted_prototypes(z_support, sy, z_q)
            dists = torch.cdist(z_q.unsqueeze(0), prototypes)  # [1, N]
            logits.append(-dists.squeeze(0))  # Negative distance

        return torch.stack(logits) 


# -----------------------------
# Prototypical Network
# -----------------------------
class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__() 
        self.backbone = backbone 

    def forward(self, sX, sy, qX):
        z_support = F.normalize(self.backbone(sX), p=2, dim=1)
        z_query = F.normalize(self.backbone(qX), p=2, dim=1)
        
        prototypes = torch.stack([
            z_support[sy == cls].mean(0) 
            for cls in torch.unique(sy)
        ])
        
        return -torch.cdist(z_query, prototypes)


# -----------------------------
# EfficientNetV2-S Backbone
# -----------------------------
efficientnet = models.efficientnet_v2_s(pretrained=True)
efficientnet.classifier = nn.Identity()
# model = PrototypicalNetworks(efficientnet).cuda()
model = AttentionPrototypicalNetwork(efficientnet).cuda()

# -----------------------------
# Mapping Data Labels
# -----------------------------
def prepare_episode_sampler(dataset):
    """Run once – returns a mapping you can reuse each episode."""
    mapping = defaultdict(list)
    
    for idx, (_, label) in tqdm(enumerate(dataset), total=len(dataset), desc="Preparing sampler"):
        cls = label.item() if torch.is_tensor(label) else label
        mapping[cls].append(idx)
    
    return mapping


train_class_to_indices = prepare_episode_sampler(train_dataset)
val_class_to_indices = prepare_episode_sampler(val_dataset)
test_class_to_indices = prepare_episode_sampler(test_dataset)

# -----------------------------
# Episode Creation
# -----------------------------
def get_fast_episode(dataset, class_to_indices, n_way=6, k_shot=3, q_queries=3,
                     device=torch.device("cpu"), return_original=False):

    selected_classes = random.sample(list(class_to_indices.keys()), n_way)
    label_map = {cls: i for i, cls in enumerate(selected_classes)}


    support_idx, query_idx = [], []
    total = k_shot + q_queries
    for cls in selected_classes:
        pool = class_to_indices[cls]
        choice = random.sample(pool, total) if len(pool) >= total else random.choices(pool, k=total)
        support_idx += choice[:k_shot]
        query_idx  += choice[k_shot:]

    # Load tensors on CPU
    sX = torch.stack([dataset[i][0] for i in support_idx])
    qX = torch.stack([dataset[i][0] for i in query_idx])
    sy = torch.tensor([label_map[
            dataset[i][1].item() if torch.is_tensor(dataset[i][1]) else dataset[i][1]
        ] for i in support_idx])
    qy = torch.tensor([label_map[
            dataset[i][1].item() if torch.is_tensor(dataset[i][1]) else dataset[i][1]
        ] for i in query_idx])

    # quick cleanup of Python lists so CUDA can actually free memory later
    del support_idx, query_idx
    if return_original:
        return sX, sy, qX, qy, selected_classes
    return sX, sy, qX, qy

#******************************
# -----------------------------
# Configuration for 15-Shot 15-Query Learning
# -----------------------------
#******************************
print("\n=============================================")
print("\n=== 15-Shot 15-Query Meta-Learning and Meta-Testing ===")
print("\n=============================================")
start_time = time.time()
start_date = datetime.now()
print(f"Started at: {start_date}")

N_SHOT = 15
SAVE_DIR = f"{N_SHOT}-shot-attention"
os.makedirs(SAVE_DIR, exist_ok=True)
# -----------------------------
# Initial evaluation before training
# -----------------------------
N_WAY = 6
N_QUERY = 15
# Initial evaluation before training
model.eval()
sX_test, sy_test, qX_test, qy_test =  get_fast_episode(test_dataset, test_class_to_indices, n_way=6, k_shot=15, q_queries=15)
sX_test, sy_test = sX_test.to(device), sy_test.to(device)
qX_test, qy_test = qX_test.to(device), qy_test.to(device)

with torch.no_grad():
    scores = model(sX_test, sy_test, qX_test)
    class_ids, _ = torch.unique(sy_test, sorted=True, return_inverse=True)
    class_to_index = {cls.item(): idx for idx, cls in enumerate(class_ids)}
    qy_indices = torch.tensor([class_to_index[c.item()] for c in qy_test], device=device)
    preds = torch.argmax(scores, dim=1)
    initial_acc = (preds == qy_indices).float().mean().item()

print(f"Initial Test Accuracy: {initial_acc:.4f}")

# -----------------------------
# 15-shot 15-query Meta-Learning
# -----------------------------
# Training loop
embedding_model = model.to(device)
base_lr         = 1e-5          # initial LR
max_lr          = 3e-4          # peak LR reached by One-Cycle
n_episodes = 15000
optimizer = torch.optim.AdamW(embedding_model.parameters(), lr=base_lr, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=n_episodes, pct_start=0.3, anneal_strategy="cos", div_factor=max_lr / base_lr)
criterion = nn.CrossEntropyLoss()

# patience = 100
# patience_counter = 0
best_val_acc = 0.0
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
lrs = []

print("==> Training the model with 15-shots...")

# Modified training loop with cached episodes
best_val_acc = 0.0
best_val_loss = 0.0
patience_counter = 0

for episode in range(n_episodes):
    # Get pre-generated episode
    sX, sy, qX, qy = get_fast_episode(train_dataset, train_class_to_indices, n_way=6, k_shot=15, q_queries=15)
    sX, sy = sX.to(device), sy.to(device)
    qX, qy = qX.to(device), qy.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass
    scores = embedding_model(sX, sy, qX)
    
    # Create episode-specific class mapping
    class_ids, _ = torch.unique(sy, sorted=True, return_inverse=True)
    class_to_index = {cls.item(): idx for idx, cls in enumerate(class_ids)}
    qy_indices = torch.tensor([class_to_index[c.item()] for c in qy], device=device)
    
    # Calculate loss
    loss = criterion(scores, qy_indices)
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Training metrics
    with torch.no_grad():
        preds = scores.argmax(dim=1)
        acc   = (preds == qy_indices).float().mean().item()

    train_losses.append(loss.item())
    train_accuracies.append(acc)
    lrs.append(scheduler.get_last_lr()[0])

    if episode % 50 == 0:
        curr_lr = scheduler.get_last_lr()[0]
        print(f"[Episode {episode:03d}] "
              f"LR {curr_lr:6.2e} | "
              f"Loss {loss.item():.4f} | "
              f"Train Acc {acc:.4f}")

    # Memory management during training
    if episode % 500 == 0 and episode > 0:
        torch.cuda.empty_cache()

    # Validation
    if episode % 50 == 0:
        embedding_model.eval()
        with torch.no_grad():
            val_sX, val_sy, val_qX, val_qy = get_fast_episode(val_dataset, val_class_to_indices, n_way=6, k_shot=15, q_queries=15)
            val_sX, val_sy = val_sX.to(device), val_sy.to(device)
            val_qX, val_qy = val_qX.to(device), val_qy.to(device)
            
            val_scores = embedding_model(val_sX, val_sy, val_qX)
            val_class_ids, _ = torch.unique(val_sy, sorted=True, return_inverse=True)
            val_class_to_index = {cls.item(): idx for idx, cls in enumerate(val_class_ids)}
            val_qy_indices = torch.tensor([val_class_to_index[c.item()] for c in val_qy], device=device)
            
            val_preds = torch.argmax(val_scores, dim=1)
            val_acc = (val_preds == val_qy_indices).float().mean().item()
            val_loss = criterion(val_scores, val_qy_indices).item()
            val_accuracies.append(val_acc)
            val_losses.append(val_loss)
            
            print(f"--> [Validation] Episode {episode} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(embedding_model.state_dict(), f"{SAVE_DIR}/best_protonet_{N_SHOT}shot.pt")
                patience_counter = 0
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     torch.save(embedding_model.state_dict(), f"{SAVE_DIR}/best_protonet_{N_SHOT}shot.pt")
            #     patience_counter = 0
            # else: 
            #     patience_counter += 1
            #     print(f" No improvement. Patience: {patience_counter}/{patience}")
            #     if patience_counter >= patience:
            #         print(" Early stopping triggered.")
            #         break
            # Clear validation tensors to free memory
            del val_sX, val_sy, val_qX, val_qy, val_scores
            torch.cuda.empty_cache()
            
        embedding_model.train()
        if episode % 50 == 0:
            torch.cuda.empty_cache()

# Save history
pd.DataFrame({
    'train_loss': train_losses,
    'train_acc': train_accuracies,
    'learning_rate': lrs
}).to_csv(f'{SAVE_DIR}/train_history_{N_SHOT}shot.csv', index=False)

pd.DataFrame({
    'val_loss': val_losses,
    'val_acc': val_accuracies
}).to_csv(f'{SAVE_DIR}/val_history_{N_SHOT}shot.csv', index=False)

print(f"Best validation accuracy: {best_val_acc:.4f}")
# print(f"Best validation Loss: {best_val_loss:.4f}")

# -----------------------------
# 15-shot 15-query Meta-Testing
# -----------------------------
# Final evaluation on test set with comprehensive metrics
print("\n=== Final Evaluation ===")
embedding_model.load_state_dict(torch.load(f"{SAVE_DIR}/best_protonet_{N_SHOT}shot.pt"))
embedding_model.eval()

test_accuracies = []
all_true = []
all_preds = []
all_probs = []
all_episode_trues = []
all_episode_preds = []
all_episode_probs = []
n_test_episodes = 2000

with torch.no_grad():
    for episode_idx in tqdm(range(n_test_episodes), desc="Test Episodes"):
        # Generate episode ensuring valid class distribution
        while True:
            try:
                sX_test, sy_test, qX_test, qy_test = get_fast_episode(test_dataset, test_class_to_indices, n_way=6, k_shot=15, q_queries=15)
                # Validate episode structure
                assert len(torch.unique(sy_test)) == N_WAY
                break
            except (ValueError, AssertionError):
                continue

        # Device transfer
        sX_test, sy_test = sX_test.to(device), sy_test.to(device)
        qX_test, qy_test = qX_test.to(device), qy_test.to(device)

        # Forward pass
        scores = embedding_model(sX_test, sy_test, qX_test)

        # Create episode-specific mapping
        class_ids, _ = torch.unique(sy_test, sorted=True, return_inverse=True)
        class_to_index = {cls.item(): idx for idx, cls in enumerate(class_ids)}
        
        # Convert query labels using episode mapping
        qy_indices = torch.tensor([class_to_index[c.item()] for c in qy_test], device=device)
        
        # Calculate metrics
        probs = F.softmax(scores, dim=1)
        preds = torch.argmax(scores, dim=1)
        
        # Store original class labels for comprehensive metrics
        true_labels = [class_ids[i].item() for i in qy_indices]
        pred_labels = [class_ids[i].item() for i in preds]
        prob_vectors = probs.cpu().numpy()

        # Store per-episode results
        all_episode_trues.append(true_labels)
        all_episode_preds.append(pred_labels)
        all_episode_probs.append(prob_vectors)

        # Accumulate overall results
        all_true.extend(true_labels)
        all_preds.extend(pred_labels)
        all_probs.extend(prob_vectors)

        # Calculate episode accuracy
        episode_acc = (preds == qy_indices).float().mean().item()
        test_accuracies.append(episode_acc)

end_time = time.time()
end_date = datetime.now()

# -----------------------------
# Utility Functions
# -----------------------------
def compute_ci(scores):
    mean = np.mean(scores)
    delta = sem(scores) * 1.96
    return mean, (mean - delta, mean + delta)

def plot_confusion(cm, classes, prefix):
    pct = cm.astype(float) / cm.sum(axis=1)[:,None]
    plt.figure(figsize=(8,6))
    sns.heatmap(pct, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (%) - {prefix}")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/confusion_{prefix}.png", dpi=300)
    plt.close()

def plot_roc_pr(y_true, y_prob, classes, prefix):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    plt.figure(figsize=(14,6))
    # ROC
    plt.subplot(1,2,1)
    for i,c in enumerate(classes):
        fpr,tpr,_ = roc_curve((y_true==i).astype(int), y_prob[:,i])
        auc = roc_auc_score((y_true==i).astype(int), y_prob[:,i])
        plt.plot(fpr,tpr,label=f"{c}(AUC={auc:.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.title("ROC Curves"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
    # PR
    plt.subplot(1,2,2)
    for i,c in enumerate(classes):
        prec,rec,_ = precision_recall_curve((y_true==i).astype(int), y_prob[:,i])
        ap = average_precision_score((y_true==i).astype(int), y_prob[:,i])
        plt.plot(rec,prec,label=f"{c}(AP={ap:.2f})")
    plt.title("Precision-Recall"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/roc_pr_{prefix}.png", dpi=300)
    plt.close()

def plot_tsne(prob_vectors, labels, class_names, prefix="5shot"):
    """
    Computes and saves a t-SNE plot of the probability vectors per-class scatter loops.
    """
    # Convert to numpy
    prob_vectors = np.array(prob_vectors)
    labels = np.array(labels)

    # Subsample for speed/clarity if needed
    if len(prob_vectors) > 2000:
        idx = np.random.choice(len(prob_vectors), 2000, replace=False)
        prob_vectors = prob_vectors[idx]
        labels = labels[idx]

    # t-SNE embedding
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    reduced = tsne.fit_transform(prob_vectors)

    # Pick Set3 colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))

    # Plot
    plt.figure(figsize=(12, 8))
    for i, (cls_name, col) in enumerate(zip(class_names, colors)):
        mask = labels == i
        plt.scatter(
            reduced[mask, 0], reduced[mask, 1],
            c=[col],
            label=cls_name,
            alpha=0.7,
            s=20
        )

    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.title(f"t-SNE Visualization of Learned Features ({prefix})", fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save to disk
    plt.savefig(f"{SAVE_DIR}/tsne_{prefix}.png", dpi=300)
    plt.close()

def calibration(y_true,y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    # Brier
    bs = np.mean([brier_score_loss((y_true==i).astype(int), y_prob[:,i])
                  for i in range(y_prob.shape[1])])
    # ECE
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred==y_true).astype(int)
    bins = np.linspace(0,1,11)
    ece=0
    for l,h in zip(bins[:-1],bins[1:]):
        mask = (conf>l)&(conf<=h)
        if mask.any(): ece+=abs(correct[mask].mean()-conf[mask].mean())*mask.mean()
    return ece, bs

def evaluate(test_acc, y_true, y_pred, y_prob, classes, prefix, episode_trues=None, episode_preds=None, episode_probs=None):
    # ensure numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    res = {}
    # accuracy + CI
    res['mean_acc'], res['ci'] = compute_ci(test_acc)
    res['std'] = np.std(test_acc)
    # Basic metrics
    res['overall_acc'] = accuracy_score(y_true, y_pred)
    res['macro_prec'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    res['macro_rec'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    res['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    res['weighted_prec'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    res['weighted_rec'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    res['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    res['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    res['mcc'] = matthews_corrcoef(y_true, y_pred)
    res['bal_acc'] = balanced_accuracy_score(y_true, y_pred)
    # Top-3 accuracy
    res['top3'] = top_k_accuracy_score(y_true, y_prob, k=3)

    # Classification report
    cr = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    for label in cr:
        if isinstance(cr[label], dict):
            for metric in cr[label]:
                if isinstance(cr[label][metric], float):
                    cr[label][metric] = round(cr[label][metric], 4)
        elif isinstance(cr[label], float):
            cr[label] = round(cr[label], 4)
    res['class_report'] = cr

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    res['confusion'] = cm

    # Specificity per class
    spec = []
    for i in range(len(classes)):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        spec.append(TN / (TN + FP) if (TN + FP) > 0 else 0)
    res['specificity'] = spec
    res['macro_specificity'] = np.mean(spec)

    # Calibration
    res['ece'], res['brier'] = calibration(y_true, y_prob)

    # Per-class ROC/PR
    auc_rocs, auc_prs, avg_precs = [], [], []
    for i in range(len(classes)):
        bin_true = (y_true == i).astype(int)
        auc_rocs.append(roc_auc_score(bin_true, y_prob[:, i]))
        prec, rec, _ = precision_recall_curve(bin_true, y_prob[:, i])
        auc_prs.append(auc(rec, prec))
        avg_precs.append(average_precision_score(bin_true, y_prob[:, i]))
    res['auc_roc'] = auc_rocs
    res['auc_pr'] = auc_prs
    res['avg_prec'] = avg_precs

    # CI for macro F1, macro AUROC, macro AUPRC
    if episode_trues and episode_preds and episode_probs:
        f1s = []
        aurocs = []
        auprcs = []
        for ep_true, ep_pred, ep_prob in zip(episode_trues, episode_preds, episode_probs):
            f1s.append(f1_score(ep_true, ep_pred, average='macro', zero_division=0))
            per_class_aurocs = []
            per_class_auprcs = []
            for i in range(len(classes)):
                bin_true = (np.array(ep_true) == i).astype(int)
                prob = np.array(ep_prob)[:, i]
                if np.sum(bin_true) > 0 and np.sum(bin_true) < len(bin_true):
                    per_class_aurocs.append(roc_auc_score(bin_true, prob))
                    per_class_auprcs.append(average_precision_score(bin_true, prob))
            if per_class_aurocs:
                aurocs.append(np.mean(per_class_aurocs))
            if per_class_auprcs:
                auprcs.append(np.mean(per_class_auprcs))

        res['f1'],res['macro_f1_ci']     = compute_ci(f1s)
        res['auroc'],res['macro_auroc_ci']  = compute_ci(aurocs)
        res['auprc'],res['macro_auprc_ci']  = compute_ci(auprcs)

    return res

# -----------------------------
# Comprehensive Metrics
# -----------------------------
res = evaluate(
    test_accuracies,
    all_true, all_preds, np.array(all_probs),
    brain_activities, f"{N_SHOT}shot",
    episode_trues=all_episode_trues,
    episode_preds=all_episode_preds,
    episode_probs=all_episode_probs
)
print("\n=============================")
print("=== Comprehensive Metrics ===")
print("====== 15-Shot 15-Query Learning ======")
print("=============================")
print(f"Final Test Accuracy over {n_test_episodes} episodes:")
print(f"Mean Accuracy: {res['mean_acc']:.4f} ± {(res['ci'][1] - res['mean_acc']):.4f} (95% CI: [{res['ci'][0]:.4f}, {res['ci'][1]:.4f}])")
print(f"Standard Deviation: {res['std']:.4f}")
print(f"F1 Score: {res['f1']:.4f} ± {(res['macro_f1_ci'][1] - res['f1']):.4f} (95% CI: [{res['macro_f1_ci'][0]:.4f}, {res['macro_f1_ci'][1]:.4f}])")
print(f"AUROC:    {res['auroc']:.4f} ± {(res['macro_auroc_ci'][1] - res['auroc']):.4f} (95% CI: [{res['macro_auroc_ci'][0]:.4f}, {res['macro_auroc_ci'][1]:.4f}])")
print(f"AUPRC:    {res['auprc']:.4f} ± {(res['macro_auprc_ci'][1] - res['auprc']):.4f} (95% CI: [{res['macro_auprc_ci'][0]:.4f}, {res['macro_auprc_ci'][1]:.4f}])")

# Basic metrics
print("\n📊 BASIC METRICS:")
print(f"Overall Accuracy: {res['overall_acc']:.4f}")
print(f"Precision (Macro): {res['macro_prec']:.4f}")
print(f"Recall (Macro): {res['macro_rec']:.4f}")
print(f"F1 Score (Macro): {res['macro_f1']:.4f}")
mean_auc_roc = np.mean(res['auc_roc'])
print(f"Mean AUC-ROC (Macro): {mean_auc_roc:.4f}")
print(f"Cohen's Kappa: {res['cohen_kappa']:.4f}")

print(f"Balanced Accuracy: {res['bal_acc']:.4f}")
print(f"Matthews Correlation Coefficient: {res['mcc']:.4f}")
print(f"Specificity (Macro): {res['macro_specificity']:.4f}")

geom_mean = np.sqrt(res['macro_prec'] * res['macro_rec'])
print(f"Geometric Mean Score: {geom_mean:.4f}")

seizure_metrics = res['class_report']['Seizure']
print("\n🔎 SEIZURE DETECTION METRICS:")
print(f"Precision: {seizure_metrics['precision']:.4f}")
print(f"Recall (Sensitivity): {seizure_metrics['recall']:.4f}")
print(f"F1 Score: {seizure_metrics['f1-score']:.4f}")
print(f"Specificity: {res['specificity'][0]:.4f}")
print(f"AUC-ROC: {res['auc_roc'][0]:.4f}")
print(f"AUC-PRC: {res['auc_pr'][0]:.4f}")
print(f"Average Precision: {res['avg_prec'][0]:.4f}")

# Calibration metrics
print("\n📈 CALIBRATION METRICS:")
print(f"Expected Calibration Error (ECE): {res['ece']:.4f}")
print(f"Brier Score: {res['brier']:.4f}")

# Macro-averaged micro and top-k
print("\n🎯 MULTI-AVERAGE METRICS:")
# Micro precision == accuracy
print(f"Precision - Micro: {res['weighted_prec']:.4f}")
print(f"Recall - Micro: {res['weighted_rec']:.4f}")
print(f"F1 Score - Micro: {res['weighted_f1']:.4f}")
print(f"Top-3 Accuracy: {res['top3']:.4f}")

# Class-wise metrics
print("\n🔍 CLASS-WISE METRICS:")
for i, cls in enumerate(brain_activities):
    m = res['class_report'][cls]
    print(f"{cls}:")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall:    {m['recall']:.4f}")
    print(f"  F1 Score:  {m['f1-score']:.4f}")
    print(f"  Specificity: {res['specificity'][i]:.4f}")
    print(f"  AUC-ROC:   {res['auc_roc'][i]:.4f}")
    print(f"  AUC-PRC:   {res['auc_pr'][i]:.4f}")
    print(f"  Avg Precision: {res['avg_prec'][i]:.4f}")

# Timing information
duration = (end_time - start_time) / 60
print("\n📅 TIMING INFORMATION:")
print(f"Start Date: {start_date}")
print(f"End Date:   {end_date}")
print(f"Total Duration: {duration:.2f} minutes")

# Classification report
print("\n📋 CLASSIFICATION REPORT:")
report_df = pd.DataFrame(res['class_report']).transpose()
print(report_df.loc[brain_activities + ['accuracy','macro avg','weighted avg']])

# Confusion matrix
print("\n🔥 CONFUSION MATRIX:")
plot_confusion(res['confusion'], brain_activities, f"{N_SHOT}shot")
cm_num = pd.DataFrame(res['confusion'], index=brain_activities, columns=brain_activities)
print(cm_num)

# ROC and PR curves
print("\n📈 ROC/PR CURVES saved to PNG files.")
plot_roc_pr(all_true, all_probs, brain_activities, f"{N_SHOT}shot")

# t-SNE
print("\n🗺️ t-SNE VISUALIZATION saved to PNG file.")
plot_tsne(all_probs, all_true, brain_activities, f"{N_SHOT}shot")

# Performance summary table
perf_summary = {
    'Metric': [
        'Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 Score (Macro)',
        'Specificity (Macro)', 'AUC-ROC (Macro)', 'Cohen Kappa', 'Balanced Accuracy',
        'Matthews Corr Coef','Top-3 Accuracy','Expected Calibration Error','Brier Score'
    ],
    'Value': [
        f"{res['overall_acc']:.4f}",
        f"{res['macro_prec']:.4f}",
        f"{res['macro_rec']:.4f}",
        f"{res['macro_f1']:.4f}",
        f"{res['macro_specificity']:.4f}",
        f"{mean_auc_roc:.4f}",
        f"{res['cohen_kappa']:.4f}",
        f"{res['bal_acc']:.4f}",
        f"{res['mcc']:.4f}",
        f"{res['top3']:.4f}",
        f"{res['ece']:.4f}",
        f"{res['brier']:.4f}"
    ]
}
perf_df = pd.DataFrame(perf_summary)
print("\n📋 PERFORMANCE SUMMARY TABLE:")
print(perf_df.to_string(index=False))
perf_df.to_csv(f"{SAVE_DIR}/performance_summary_{N_SHOT}shot.csv", index=False)

# Class-wise performance table
class_perf = pd.DataFrame({
    'Classes':   brain_activities,
    'Precision': [res['class_report'][c]['precision'] for c in brain_activities],
    'Recall':    [res['class_report'][c]['recall']    for c in brain_activities],
    'F1-Score':  [res['class_report'][c]['f1-score']  for c in brain_activities],
    'Specificity':[res['specificity'][i]             for i in range(len(brain_activities))],
    'AUC-ROC':   [res['auc_roc'][i]                 for i in range(len(brain_activities))],
    'AUC-PRC':   [res['auc_pr'][i]                  for i in range(len(brain_activities))],
    'Avg Precision': [res['avg_prec'][i]            for i in range(len(brain_activities))]
}, index=brain_activities)
class_perf = class_perf.round(4)

print("\n📊 CLASS-WISE PERFORMANCE TABLE:")
print(class_perf)
class_perf.to_csv(f"{SAVE_DIR}/classwise_performance_{N_SHOT}shot.csv", index=False)

print(f"✅ Done. All outputs in '{SAVE_DIR}' directory")

print("Cleaning up GPU memory...")

# Delete all large objects/tensors/outputs created during any phase
all_objs = [
    'embedding_model', 'model', 'optimizer',
    'train_dataset', 'val_dataset', 'test_dataset',
    'sX', 'sy', 'qX', 'qy',
    'sX_test', 'sy_test', 'qX_test', 'qy_test',
    'val_sX', 'val_sy', 'val_qX', 'val_qy',
    'val_scores', 'scores', 'preds', 'qy_indices',
    'class_ids', 'class_to_index', 'val_class_ids', 'val_class_to_index',
    'val_qy_indices', 'val_preds',
    'all_true', 'all_preds', 'all_probs',
    'all_episode_trues', 'all_episode_preds', 'all_episode_probs'
]

for obj_name in all_objs:
    if obj_name in globals():
        del globals()[obj_name]

gc.collect()
torch.cuda.empty_cache()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
print("Cleanup complete.")
