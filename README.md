# ProtoEEG-QA: Attention-Guided Few-Shot Prototypical Network for ICU EEG Abnormal Pattern Recognition        
---   
This repository contains the complete implementation of **ProtoEEG** and its attention-augmented variant **ProtoEEG-QA** for few-shot classification of ACNS-defined EEG abnormalities in neurocritical care. The framework emphasizes interpretability, calibration, and efficiency—enabling clinically trustworthy detection in resource-constrained environments.
---

## 📂 Contents

- `ProtoEEG.py`: Baseline few-shot prototypical network
- `ProtoEEG-QA.py`: Query-aware attention-enhanced model
- `models/`: Pretrained weights
- `figures/`: Confusion matrices, t-SNE visualizations
- `results/`: Metrics for 5-, 10-, 15-shot evaluations
- `README.md`: Reproducibility guide and detailed documentation

---
## 🧪 Reproducibility

1. Clone repository
git clone https://gitfront.io/r/deep/CpM9t1JqZZfR/ProtoEEG-QA
cd ProtoEEG-QA

2. Install dependencies
pip install -r requirements.txt

3. Run ProtoEEG
python ProtoEEG.py 

4. Run ProtoEEG-QA
python ProtoEEG-QA.py 


---
## Appendix A -- Implementation Details
- **Dataset**: [HMS Harmful Brain Activity Classification dataset](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data)

### 🧪 Hyperparameters & Rationale
The following hyperparameters were selected for effective meta-learning and reliable model convergence:

| Parameter              | Value                          | Rationale                                                  |
|------------------------|--------------------------------|------------------------------------------------------------|
| Backbone               | EfficientNetV2-S               | Lightweight yet powerful spectrogram encoder               |
| Embedding Dimension    | 1280                           | Matches EfficientNetV2-S output                           |
| Support Shots (K)      | 5, 10, 15                      | Evaluate generalization under low supervision              |
| Query Samples (Q)      | 15                             | Fixed for consistent meta-test episodes                    |
| Training Episodes      | 15,000                         | Episodic diversity and convergence                         |
| Test Episodes          | 2,000                          | Robust statistics across random episodes                   |
| Optimizer              | AdamW                          | Combines adaptive learning with weight decay               |
| Learning Rate Range    | 1e-5 to 3e-4                   | Used with OneCycleLR scheduler                             |
| Scheduler              | OneCycleLR                     | Encourages fast convergence with warm-up                   |
| Weight Decay           | 1e-3                           | Regularization to prevent overfitting                      |
| Loss Function          | CrossEntropy                   | Multi-class episodic classification                        |
| Attention MLP          | [512 → 64 → softplus]          | Lightweight attention scoring per support-query pair       |

---
# Appendix B -- 📊 Full Evaluation Results
### Additional Results- Class-Wise Performance 

| Model         | Shot    | Class   | Prec.  | Rec.   | F1     | Spec.  | AUROC  | AUPRC  |
|---------------|---------|---------|--------|--------|--------|--------|--------|--------|
| **ProtoEEG**  | 5-shot  | Seizure | 0.8379 | 0.8373 | 0.8376 | 0.9676 | 0.9354 | 0.8599 |
|               |         | GPD     | 0.8309 | 0.8351 | 0.8330 | 0.9660 | 0.9353 | 0.8581 |
|               |         | LRDA    | 0.8351 | 0.8333 | 0.8342 | 0.9671 | 0.9338 | 0.8555 |
|               |         | Other   | 0.8406 | 0.8329 | 0.8368 | 0.9684 | 0.9351 | 0.8589 |
|               |         | GRDA    | 0.8330 | 0.8370 | 0.8350 | 0.9664 | 0.9357 | 0.8580 |
|               |         | LPD     | 0.8338 | 0.8357 | 0.8348 | 0.9667 | 0.9342 | 0.8557 |
|               | 10-shot | Seizure | 0.8464 | 0.8504 | 0.8484 | 0.9691 | 0.9367 | 0.8753 |
|               |         | GPD     | 0.8483 | 0.8490 | 0.8486 | 0.9696 | 0.9375 | 0.8777 |
|               |         | LRDA    | 0.8534 | 0.8474 | 0.8504 | 0.9709 | 0.9360 | 0.8739 |
|               |         | Other   | 0.8478 | 0.8500 | 0.8489 | 0.9695 | 0.9391 | 0.8798 |
|               |         | GRDA    | 0.8447 | 0.8466 | 0.8457 | 0.9689 | 0.9340 | 0.8717 |
|               |         | LPD     | 0.8511 | 0.8481 | 0.8496 | 0.9703 | 0.9337 | 0.8725 |
|               | 15-shot | Seizure | 0.8428 | 0.8367 | 0.8397 | 0.9688 | 0.9300 | 0.8540 |
|               |         | GPD     | 0.8368 | 0.8369 | 0.8369 | 0.9674 | 0.9304 | 0.8534 |
|               |         | LRDA    | 0.8307 | 0.8345 | 0.8326 | 0.9660 | 0.9273 | 0.8477 |
|               |         | Other   | 0.8434 | 0.8361 | 0.8397 | 0.9689 | 0.9293 | 0.8528 |
|               |         | GRDA    | 0.8333 | 0.8368 | 0.8350 | 0.9665 | 0.9296 | 0.8525 |
|               |         | LPD     | 0.8327 | 0.8385 | 0.8356 | 0.9663 | 0.9303 | 0.8537 |
| **ProtoEEG-QA** | 5-shot  | Seizure | 0.8497 | 0.8520 | **0.8508** | 0.9699 | 0.9697 | 0.9175 |
|               |         | GPD     | 0.8444 | 0.8544 | **0.8494** | 0.9685 | 0.9696 | 0.9191 |
|               |         | LRDA    | 0.8549 | 0.8539 | **0.8544** | 0.9710 | 0.9695 | 0.9191 |
|               |         | Other   | 0.8515 | 0.8492 | **0.8503** | 0.9704 | 0.9690 | 0.9178 |
|               |         | GRDA    | 0.8511 | 0.8519 | **0.8515** | 0.9702 | 0.9694 | 0.9181 |
|               |         | LPD     | 0.8623 | 0.8473 | **0.8548** | 0.9728 | 0.9698 | 0.9176 |
|               | 10-shot | Seizure | 0.8491 | 0.8511 | **0.8501** | 0.9697 | 0.9698 | 0.9194 |
|               |         | GPD     | 0.8523 | 0.8567 | **0.8545** | 0.9708 | 0.9700 | 0.9207 |
|               |         | LRDA    | 0.8493 | 0.8523 | **0.8508** | 0.9695 | 0.9699 | 0.9200 |
|               |         | Other   | 0.8535 | 0.8557 | **0.8546** | 0.9709 | 0.9695 | 0.9197 |
|               |         | GRDA    | 0.8555 | 0.8564 | **0.8559** | 0.9715 | 0.9701 | 0.9204 |
|               |         | LPD     | 0.8584 | 0.8499 | **0.8541** | 0.9717 | 0.9694 | 0.9194 |
|               | 15-shot | Seizure | 0.8544 | 0.8521 | **0.8533** | 0.9714 | 0.9729 | 0.9169 |
|               |         | GPD     | 0.8565 | 0.8592 | **0.8578** | 0.9720 | 0.9730 | 0.9172 |
|               |         | LRDA    | 0.8578 | 0.8572 | **0.8575** | 0.9722 | 0.9726 | 0.9177 |
|               |         | Other   | 0.8580 | 0.8591 | **0.8585** | 0.9723 | 0.9731 | 0.9173 |
|               |         | GRDA    | 0.8592 | 0.8604 | **0.8598** | 0.9725 | 0.9729 | 0.9170 |
|               |         | LPD     | 0.8597 | 0.8586 | **0.8591** | 0.9726 | 0.9733 | 0.9174 |

---
#Appendix C
## 🔍 Visual Comparison: ProtoEEG vs. ProtoEEG-QA (5-shot, 10-shot, 15-shot)

This section illustrates the post-training visualization results of **ProtoEEG** (baseline) and **ProtoEEG-QA** (attention-enhanced) models across three different few-shot learning scenarios: **5-shot**, **10-shot**, and **15-shot**. Each row compares the two models using:

- **t-SNE plots** to visualize learned feature embeddings  
- **Confusion matrices** to show class-wise performance  
- **ROC & PR curves** to assess discriminative confidence and class separability

### 🧠 Interpretability

- ProtoEEG-QA links each prediction to support embeddings via learned attention scores.
- Facilitates visual traceability for clinicians
- Explains classification via ranked reference examples

---

### 🧪 5-Shot Setting

#### ProtoEEG vs ProtoEEG-QA (5-shot)

<table>
  <tr>
    <td align="center"><b>ProtoEEG</b><br/><em>t-SNE</em><br/>
      <img src="https://raw.githubusercontent.com/Deepak-Mewada/ProtoEEG_additional_Material/main/figures/tsne_5shot_vanilla.png" width="500"/>
    </td>
    <td align="center"><b>ProtoEEG-QA</b><br/><em>t-SNE</em><br/>
      <img src="https://raw.githubusercontent.com/Deepak-Mewada/ProtoEEG_additional_Material/main/figures/tsne_5shot_attention.png" width="500"/>
    </td>
  </tr>
  <tr>
    <td align="center"><em>Confusion Matrix</em><br/>
      <img src="https://raw.githubusercontent.com/Deepak-Mewada/ProtoEEG_additional_Material/main/figures/confusion_5shot_vanilla.png" width="500"/>
    </td>
    <td align="center"><em>Confusion Matrix</em><br/>
      <img src="https://raw.githubusercontent.com/Deepak-Mewada/ProtoEEG_additional_Material/main/figures/confusion_5shot_attention.png" width="500"/>
    </td>
  </tr>
  <tr>
    <td align="center"><em>ROC & PR Curves</em><br/>
      <img src="https://raw.githubusercontent.com/Deepak-Mewada/ProtoEEG_additional_Material/main/figures/rocpr_5shot_vanilla.png" width="500"/>
    </td>
    <td align="center"><em>ROC & PR Curves</em><br/>
      <img src="https://raw.githubusercontent.com/Deepak-Mewada/ProtoEEG_additional_Material/main/figures/rocpr_5shot_attention.png" width="500"/>
    </td>
  </tr>
</table>

---

### 🧪 10-Shot Setting

#### ProtoEEG vs ProtoEEG-QA (10-shot)

<table>
  <tr>
    <td align="center"><b>ProtoEEG</b><br/><em>t-SNE</em><br/>
      <img src="https://raw.githubusercontent.com/Deepak-Mewada/ProtoEEG_additional_Material/main/figures/tsne_10shot_vanilla.png" width="500"/>
    </td>
    <td align="center"><b>ProtoEEG-QA</b><br/><em>t-SNE</em><br/>
      <img src="https://raw.githubusercontent.com/Deepak-Mewada/ProtoEEG_additional_Material/main/figures/tsne_10shot_attention.png" width="500"/>
    </td>
  </tr>
  <tr>
    <td align="center"><em>Confusion Matrix</em><br/>
      <img src="https://raw.githubusercontent.com/Deepak-Mewada/ProtoEEG_additional_Material/main/figures/confusion_10shot_vanilla.png" width="500"/>
    </td>
    <td align="center"><em>Confusion Matrix</em><br/>
      <img src="https://raw.githubusercontent.com/Deepak-Mewada/ProtoEEG_additional_Material/main/figures/confusion_10shot_attention.png" width="500"/>
    </td>
  </tr>
  <tr>
    <td align="center"><em>ROC & PR Curves</em><br/>
      <img src="https://raw.githubusercontent.com/Deepak-Mewada/ProtoEEG_additional_Material/main/figures/rocpr_10shot_vanilla.png" width="500"/>
    </td>
    <td align="center"><em>ROC & PR Curves</em><br/>
      <img src="https://raw.githubusercontent.com/Deepak-Mewada/ProtoEEG_additional_Material/main/figures/rocpr_10shot_attention.png" width="500"/>
    </td>
  </tr>
</table>

---

### 🧪 15-Shot Setting

#### ProtoEEG vs ProtoEEG-QA (15-shot)

<table>
  <tr>
    <td align="center"><b>ProtoEEG</b><br/><em>t-SNE</em><br/>
      <img src="https://raw.githubusercontent.com/Deepak-Mewada/ProtoEEG_additional_Material/main/figures/tsne_15shot.png" width="500"/>
    </td>
    <td align="center"><b>ProtoEEG-QA</b><br/><em>t-SNE</em><br/>
      <img src="https://raw.githubusercontent.com/Deepak-Mewada/ProtoEEG_additional_Material/main/figures/tsne_15shot_atten.png" width="500"/>
    </td>
  </tr>
  <tr>
    <td align="center"><em>Confusion Matrix</em><br/>
      <img src="https://raw.githubusercontent.com/Deepak-Mewada/ProtoEEG_additional_Material/main/figures/confusion_15shot.png" width="500"/>
    </td>
    <td align="center"><em>Confusion Matrix</em><br/>
      <img src="https://raw.githubusercontent.com/Deepak-Mewada/ProtoEEG_additional_Material/main/figures/confusion_15shot_atten_ang.png" width="500"/>
    </td>
  </tr>
  <tr>
    <td align="center"><em>ROC & PR Curves</em><br/>
      <img src="https://raw.githubusercontent.com/Deepak-Mewada/ProtoEEG_additional_Material/main/figures/roc_pr_15shot.png" width="500"/>
    </td>
    <td align="center"><em>ROC & PR Curves</em><br/>
      <img src="https://raw.githubusercontent.com/Deepak-Mewada/ProtoEEG_additional_Material/main/figures/roc_pr_15shot_atten.png" width="500"/>
    </td>
  </tr>
</table>

---



---
