## **Title:** Multi-class classification of plant leaf diseases using a hybrid deep neural transformer system and explainable AI techniques

**Authors:** Saravanan Srinivasan et al. (2026), _Scientific Reports_ 16:18161

---

### 1. Limitations / Research Gaps

The paper is fairly transparent about its own limitations, which the authors explicitly acknowledge in the Conclusion:

- **Computational cost:** The hybrid ResViT-152 has ~92M parameters and is more expensive to train/run than lightweight models (MobileNetV3-Large, EfficientNet-Lite), making it unsuitable for edge/resource-constrained deployment as-is.
- **Controlled/semi-controlled data only:** Both datasets (D-I and D-II) are curated Kaggle datasets, not truly field-collected images. They don't capture real farming conditions such as complex backgrounds, variable lighting, occlusion, or **multi-disease co-occurrence on a single leaf**.
- **No temporal/contextual data:** The model works purely on static images — no consideration of disease progression over time or environmental/contextual metadata (weather, soil, etc.).
- **No real-world/edge deployment testing:** The study stops at benchmark evaluation; no validation on real-world datasets like PlantDoc, and no real-time/on-device deployment testing.
- **XAI validation is limited:** The 93.5% "XAI interpretability score" is not rigorously grounded — the paper itself notes that quantitative IoU-based heatmap scoring, expert agronomist validation, and failure-case analysis are left as future work rather than being done here.
- **Gap relative to literature (addressed by this paper, but worth noting as a broader field gap):** Prior works (refs 17, 18, 22, 26 etc.) were mostly single-dataset, single-crop, or lacked interpretability — the authors position ResViT-152 as closing these gaps, but their own solution still doesn't fully close the "real-world deployment" gap.

### 2. Dataset Used

Two large, publicly available Kaggle datasets covering **corn, tomato, and potato** leaves across **18 disease classes**:

| Dataset                                                  | Images | Description                                                                               |
| -------------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------- |
| **D-I** ("New Plant Diseases Dataset", Vipooool, Kaggle) | 39,203 | Lab-style images, controlled lighting, uniform backgrounds                                |
| **D-II** ("PlantifyDR Dataset", Lavaman151, Kaggle)      | 65,565 | Field-acquired images, variable lighting, cluttered backgrounds, mixed camera resolutions |

Split: stratified 70% train / 15% validation / 15% test per class. Four evaluation protocols were used: IntraTest1 (train/test D-I), IntraTest2 (train/test D-II), CrossTest1 (train D-I → test D-II), CrossTest2 (train D-II → test D-I).
### 3. Repository / Access

- **Publisher page (open access):** https://www.nature.com/articles/s41598-026-48103-3
- **DOI:** https://doi.org/10.1038/s41598-026-48103-3
- **Datasets:**
  - D-I: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
  - D-II: https://www.kaggle.com/datasets/lavaman151/plantifydr-dataset
- **Code/model repository:** Not mentioned anywhere in the paper — no GitHub link or code availability statement is given. The "Data availability" section only states datasets are available from the corresponding author on reasonable request, which is odd given both are public Kaggle datasets already cited.

### 4. Remarks / Comments

- **Strong empirical results, but framed close to the ceiling:** Accuracies in the 95–99% range across both intra- and cross-dataset tests are very high; combined with claimed near-perfect Cohen's Kappa (>0.98) and tight cross-validation variance, the results look almost "too clean," especially since disease classes are described as visually similar. This warrants some skepticism until independently reproduced.
- **Cross-dataset testing is a genuine methodological strength** relative to much of the cited prior work, which the authors correctly point out mostly evaluates on a single dataset (usually PlantVillage-derived).
- **The ablation study (Table 5) is informative** and shows each component (Transformer branch, CNN branch, fusion layer, CBAM, augmentation, AdamW) contributes incrementally — useful for understanding _why_ the hybrid works, not just _that_ it works.
- **"Data availability" statement is inconsistent** with the fact that both datasets are openly hosted on Kaggle and cited by URL in the references — this should probably just link directly rather than requiring a request to the corresponding author.
- **No code release** is a meaningful gap for reproducibility, especially for a fairly complex 92M-parameter hybrid architecture with many training details (CutMix/MixUp parameters, CBAM placement, fusion dimensions) that would benefit from a reference implementation.
- **The explainability claims (Grad-CAM++) are qualitative**, illustrated with a handful of example heatmaps (Fig. 14) rather than a systematic, quantitative validation (e.g., IoU against expert-annotated lesion masks) — the authors admit this and defer it to future work, which is a fair but notable caveat given how prominently "explainable AI" features in the title.
- Overall, this is a solid **engineering/benchmarking contribution** (a well-designed hybrid CNN-Transformer with attention and good evaluation protocol) rather than a fundamentally novel architectural or theoretical advance — the individual components (ResNet152V2, ViT, CBAM, Grad-CAM++, CutMix/MixUp) are all established techniques being combined and benchmarked rather than newly invented.

**Q1. What problem do the authors address and why is it important?**
The paper addresses unreliable, slow, and inconsistent manual detection of leaf diseases in corn, tomato, and potato, three globally important food crops, where existing DL approaches either fail to generalize across datasets, lack interpretability, or use single architectures (pure CNN or pure Transformer) that cannot jointly capture fine-grained local lesion textures and global leaf-level context; solving this matters because early, accurate, and trustworthy disease detection can reduce crop losses and support precision agriculture at scale.

**Q2. What data is used (source, size, timeframe, splits, collection process, ethics or consent)?**
Two public Kaggle datasets are used: D-I ("New Plant Diseases Dataset," 39,203 lab-style images with controlled lighting/uniform backgrounds) and D-II ("PlantifyDR Dataset," 65,565 field-acquired images with variable lighting, cluttered backgrounds, and mixed camera resolutions), together covering 18 disease classes across corn, tomato, and potato; each class was split via stratified 70% train/15% validation/15% test partitioning, no timeframe or collection process details are given beyond their Kaggle origin, and no ethics/consent statement is provided since these are pre-existing public image datasets.

**Q3. What features or inputs are used, and how were they selected or engineered?**
Inputs are raw RGB leaf images resized to 224×224 and normalized using ImageNet statistics; feature engineering includes brightness correction, contrast normalization, and color/sharpness enhancement to highlight disease-related discoloration and texture, plus online data augmentation during training (horizontal/vertical flips, ±30° rotation, brightness/contrast jitter, Gaussian noise, CutMix, and MixUp) to boost minority-class representation and reduce overfitting/dataset-specific memorization; the model itself learns hierarchical CNN features (local lesion textures) and Transformer-derived global contextual features, refined further by a CBAM attention module before fusion.

**Q4. What methods or models are applied, and what is the overall pipeline?**
The core contribution is ResViT-152, a hybrid model fusing a ResNet152V2 CNN backbone (extracting a 7×7×2048 local feature map) with a ViT-style Transformer encoder (6 self-attention blocks, 8 heads) that processes patch-embedded tokens from the CNN map to capture global context; CBAM attention refines CNN features before a fusion layer concatenates and projects both streams into a unified 512-dimensional embedding, which is classified via a dense softmax head; the pipeline includes preprocessing/augmentation, CNN+Transformer feature extraction, CBAM refinement, feature fusion, classification, training with AdamW optimizer (label smoothing, dropout, early stopping), and final Grad-CAM++ based explainability visualization.

**Q5. What baselines are used for comparison, and why were they chosen?**
Baselines include InceptionNetV3, ResNet152V2, ViT, and BERT-Vision (as representative widely-used CNN and Transformer architectures for image classification), plus lightweight/efficient models (MobileNetV3-Large, EfficientNet-Lite/EfficientNetV2+Transformer, DenseNet201, Swin Transformer-Tiny, ConvNeXt-Small) and optimizer variants (SGD, RMSProp, Adam, AdamW) for ablation; these were chosen to represent the spectrum from pure CNNs to pure Transformers to hybrid/lightweight models, enabling a fair comparison of accuracy, efficiency, robustness, and interpretability trade-offs against the proposed hybrid.

**Q6. How is performance evaluated (metrics, experimental setup, statistical tests, user studies if applicable)?**
Evaluation uses accuracy, F1-score, precision, recall, specificity, G-mean, Cohen's Kappa, ROC-AUC, and precision-recall curves, computed under four protocols (IntraTest1/2 for within-dataset performance on D-I/D-II, CrossTest1/2 for bidirectional cross-dataset generalization); five-fold cross-validation quantifies variance/stability, McNemar's test checks statistical significance of ResViT-152's superiority over baselines (p<0.05), and 95% confidence intervals confirm non-overlapping performance ranges; no human/user study was conducted, though the paper mentions expert agronomist validation of XAI outputs as a stated future direction rather than something actually performed.

**Q7. What are the key results with numbers, and how do they compare to baselines or prior work?**
ResViT-152 achieves IntraTest accuracies of 99.12% (corn), 98.94% (tomato), 99.06% (potato) on D-I and ~99.23%/98.97%/98.98% on D-II, with CrossTest1 accuracies of 96.27%/95.14%/95.06% and CrossTest2 of 95.77%/96.22%/96.15%; it outperforms all baselines with the highest overall F1-score (99.18%), robustness score (94.8%), lowest stability deviation (0.008), and highest XAI interpretability score (93.5%), also beating recent SOTA models like Ensemble Heterogeneous Transformer (98.14%), Capsule Attention Network (97.62%), and SE-ResNet152 (97.85%) on comparable tasks.

**Q8. What are the limitations and potential biases?**
Key limitations include high computational/parameter cost (~92M params) unsuitable for edge deployment; reliance on controlled/semi-controlled Kaggle datasets that don't reflect real-world field conditions (complex backgrounds, occlusion, multi-disease co-occurrence); no temporal or contextual (e.g., weather) data considered; no real-time or low-resource deployment testing; potential image bias from lighting, camera quality, and class imbalance differences between D-I and D-II (partially mitigated via stratified splitting, label smoothing, and CutMix/MixUp); and XAI validation that remains qualitative/illustrative rather than rigorously quantified.

**Q9. Is code, data, or other artifacts available to enable replication?**
The datasets (D-I and D-II) are publicly available on Kaggle with direct links provided in the references, but the paper's own "Data availability" statement instead directs readers to request data from the corresponding author, which is inconsistent; no code, trained model weights, or GitHub repository is mentioned anywhere in the paper, limiting full reproducibility of the ResViT-152 implementation.

---

## **Title:** Robust multiclass classification of crop leaf diseases using hybrid deep learning and Grad-CAM interpretability

---

**1. Lacking of the Paper / Research Gap:** The paper's stated limitations include heavy reliance on image-based inputs alone, which makes it vulnerable to real-world variations in lighting, background clutter, and image quality; the dataset covers only three crops (banana, cherry, tomato), so generalizability to other plants or rarer diseases is untested; the Hybrid ConvNet-ViT model requires a moderate computational footprint (~32–35M parameters, ~9–12 GFLOPs), which could hinder deployment on low-resource/edge devices without further optimization; the system performs classification only, with no disease severity estimation or treatment recommendation; and all experiments were run on a single internally-split dataset (via 5-fold CV), so cross-dataset generalization (e.g., to PlantDoc or PlantVillage) remains unverified — the authors explicitly flag this as future work.

**2. Dataset Used:** A publicly released dataset (hosted on Kaggle, ref. 30: "bct-image-dataset" by sankarlalmurugesan) containing labeled banana, cherry, and tomato leaf images across 9 classes — BH, BU, CH, CPM (banana/cherry-related), and five tomato classes: TSL (Septoria Leaf Spot), TSM (Spider Mites), TTS (Target Spot), TTM (Mosaic Virus), and TTY (Yellow Leaf Curl Virus). The data was split 70/15/15 (train/val/test), with class sizes ranging from the smallest (BH: 109 train/23 val/23 test) to the largest (TTY: 3,750 train/804 val/804 test) — a notably imbalanced distribution.

**3. Repository / Summary Source:** The paper itself doesn't mention a code repository (no GitHub/artifact link given); it only states the dataset is available from the corresponding author on reasonable request, and the dataset is sourced from the Kaggle link in reference 30. No mention of released model weights or code for the Hybrid ConvNet-ViT implementation.

**4. Remarks / Comments:** A few things worth flagging for your review: the paper's Related Work table (Table 1) and its surrounding text oddly reference "drowsiness detection" methods and limitations (e.g., eye closure, PERCLOS, blink detection) — this appears to be a copy-paste error from an unrelated manuscript template and is inconsistent with the plant-disease framing. Also, the introduction's dataset class list is garbled ("Classes: BH, BU, CH, and CPM. Five categories are planted, placed..."), reading like a drafting artifact rather than clean prose. The reported accuracy gain from the Hybrid model over standalone ViT is modest (99.29% vs. 98.92%, a 0.37-point gain) for a 3× cost in FLOPs versus ConvNeXt, so the practical justification leans more on robustness/Grad-CAM interpretability and cross-validation stability than raw accuracy — this is honestly acknowledged in the ablation discussion, which is a strength. The confusion matrices are also explicitly described as "accurately simulated" from reported accuracy values and class sizes rather than being raw outputs of an actual run — worth noting if you're citing this as empirical evidence rather than a modeled reconstruction. Confidence intervals (Table 12) are a nice statistical touch for class-wise reliability, particularly useful given the class imbalance (e.g., BH/BU/TTM have very small test sets and correspondingly wide CIs).

**Q1. What problem do the authors address and why is it important?**
The authors address the challenge of accurately detecting and classifying foliar diseases across multiple crop species — banana, cherry, and tomato — using a single unified deep learning model. This matters because foliar diseases (e.g., Black Sigatoka in banana, Cherry Leaf Spot, Early/Late Blight in tomato) significantly reduce photosynthetic capacity, weaken plant health, and threaten crop yield and food security. Traditional visual diagnosis by experts is slow, subjective, inconsistent, and often inaccessible in rural or under-resourced farming regions, especially since symptoms overlap across diseases and even with non-disease stresses like nutrient deficiency. Existing deep learning approaches also tend to be crop-specific or architecturally limited — ConvNets capture local texture/edge patterns well but miss global context, while transformers capture long-range dependencies but lack the inductive biases needed for fine-grained local symptom detection. The paper aims to close this gap with a hybrid model that generalizes across multiple crop types without species-specific retraining.

**Q2. What data is used (source, size, timeframe, splits, collection process, ethics or consent)?**
The dataset is a publicly released collection (hosted on Kaggle, referenced as "bct-image-dataset") of healthy and diseased banana, cherry, and tomato leaf images spanning 9 classes: BH, BU, CH, CPM, and five tomato disease classes (TSL, TSM, TTS, TTM, TTY). Total class sizes range from 155 images (BH) to 5,358 images (TTY). The data was split 70% train / 15% validation / 15% test, e.g., TTY had 3,750 train/804 val/804 test images, while BH had only 109 train/23 val/23 test images — a fairly imbalanced distribution across classes. No specific timeframe, collection methodology, licensing terms, or consent/ethics details are mentioned in the paper; it simply cites the Kaggle source and states that data is available from the corresponding author "on reasonable request."

**Q3. What features or inputs are used, and how were they selected or engineered?**
The raw input is RGB leaf images, uniformly resized to 224×224×3 (some models like transformers also referenced 384×384 as an option) using bilinear interpolation to standardize inputs across all architectures with differing native requirements. Rather than hand-crafted features, the models learn feature representations automatically: ConvNet backbones (EfficientNetV2, ConvNeXt) extract local texture/edge features via convolutions, while transformer-based models (ViT, Swin) partition images into fixed 16×16 patches, flatten them into token embeddings, and add positional encodings to preserve spatial relationships. Data augmentation was used to enrich inputs and improve generalization: random cropping, horizontal flipping (p=0.5), and random rotation (±15°). Images were normalized per-channel using ImageNet mean/std statistics ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]) to enable transfer learning from pretrained weights and stabilize training convergence.

**Q4. What methods or models are applied, and what is the overall pipeline?**
Five models were evaluated: EfficientNetV2, ConvNeXt, Swin Transformer, ViT, and the proposed Hybrid ConvNet-ViT. The pipeline (Fig. 1) proceeds as: input images → preprocessing (resize, normalize, augment) → dataset split (70/15/15) → model training with 5-fold cross-validation → output evaluation (sensitivity, specificity, accuracy, F1). The proposed Hybrid model specifically begins with a convolutional stem (Conv-BatchNorm-ReLU blocks) that extracts low/mid-level spatial features, which are then flattened and linearly projected into token embeddings; positional embeddings are added, and the sequence passes through a transformer encoder (6 layers, 8 heads, embedding dim 768, MLP hidden dim 3072, dropout 0.1) using multi-head self-attention with residual connections and layer normalization. A global average pooling step converts encoder output into a single vector, which feeds a fully connected classification head with dropout, followed by softmax for final class probabilities. Training used the AdamW optimizer (learning rate 0.0005 for the hybrid model), batch size 64, over 50 epochs, with a fixed random seed for reproducibility.

**Q5. What baselines are used for comparison, and why were they chosen?**
Four pretrained baselines were selected: EfficientNetV2 (chosen for high accuracy at low computational cost via compound scaling), ConvNeXt (a modernized ConvNet with transformer-inspired design elements like LayerNorm and GELU activations, chosen for strong hierarchical feature extraction), Swin Transformer (chosen for its hierarchical shifted-window attention that balances local and global feature capture efficiently), and ViT (chosen for its pure self-attention mechanism capturing long-range dependencies). These four represent the current spectrum of architectural paradigms — pure ConvNets, modernized ConvNets, and pure/hierarchical transformers — providing broad coverage to contextualize the hybrid model's advantages. Additionally, in the discussion section, the authors compare against external state-of-the-art models from prior literature (VGG19, MobileNet ConvNet, FCDConvNet, EfficientNetB0, Hybrid ConvNet, ToLeD, RConvNet, SNDPN) to position their model against the broader field.

**Q6. How is performance evaluated (metrics, experimental setup, statistical tests, user studies if applicable)?**
Five metrics were used: Accuracy, Sensitivity/Recall, Specificity, Precision, and F1-score (formulas given in Eqs. 7–11). Evaluation occurred across three phases: training (tracked loss/accuracy every 10 epochs up to 50), 5-fold cross-validation (dataset split into 5 folds, each metric averaged across folds), and final testing on the held-out 15% test set. Experiments ran on a Windows 11 system with 16GB RAM and 1TB SSD, implemented in PyTorch with NumPy, scikit-learn, OpenCV, and Matplotlib. For statistical rigor, the authors computed 95% confidence intervals for class-wise precision, recall, and F1-score using non-parametric bootstrapping with 1,000 resamples on test predictions, to quantify variability and rule out overfitting or data leakage as explanations for high scores. An ablation study also tested model variants (individual backbones, 2-/3-branch combinations, ViT head removal, frozen-layer ratios, learning rate variations) to isolate each component's contribution. Grad-CAM visualizations were used qualitatively to assess model interpretability rather than as a formal user study.

**Q7. What are the key results with numbers, and how do they compare to baselines or prior work?**
The proposed Hybrid ConvNet-ViT achieved the best testing accuracy of 99.29%, with sensitivity 99.12%, specificity 99.22%, precision 99.15%, and F1-score 99.18% — outperforming EfficientNetV2 (98.85%), ConvNeXt (98.92%), Swin Transformer (98.75%), and ViT (98.92%) in testing accuracy. In 5-fold cross-validation, the hybrid model averaged 99.22% accuracy, 99.08% sensitivity, 99.17% specificity, 99.13% precision, and 99.10% F1, again exceeding all baselines (which ranged from 98.21%–98.51% average accuracy). Training-phase results showed the hybrid model reaching 99.29% accuracy and 0.13 loss by epoch 50, consistently ahead of other models at every checkpoint. Against external SOTA models in the literature, the hybrid model (99.29%) also outperformed VGG19 (99.16%), MobileNet ConvNet (98.60%), FCDConvNet (98.02%), SNDPN (97.59%), RConvNet (96.73%), ConvNet (95.81%), ToLeD (91.21%), Hybrid ConvNet (91.17%), and EfficientNetB0 (87.83%). The ablation study showed the full hybrid configuration (99.29% acc, 99.18% F1, 33M params, 12 GFLOPs) beat all partial combinations (e.g., EffNet+ConvNeXt+ViT: 98.72%; all-ConvNet-branches-no-ViT: 98.02%) and variants with the ViT head removed (98.51%) or altered learning rates/frozen layers.

**Q8. What are the limitations and potential biases?**
The authors explicitly note several limitations: reliance solely on image-based inputs makes the model sensitive to lighting variation, background clutter, and image quality issues common in real field conditions; the dataset covers only three crop types, so generalizability to other plants or less common diseases is untested; the model's moderate computational footprint (32–35M parameters, up to 12 GFLOPs) could be a barrier to deployment on low-resource or edge devices without further optimization; the approach only classifies disease presence/type and does not estimate severity or suggest treatments; and all experiments derive from a single internally-split dataset, so despite 5-fold cross-validation ensuring internal robustness, cross-dataset generalizability (e.g., to PlantDoc or PlantVillage) remains untested and is flagged as future work. A potential bias not explicitly discussed by the authors is class imbalance — e.g., TTY has ~34× more samples than BH — which could affect minority-class reliability despite the reported high aggregate accuracy (reflected in wider confidence intervals for smaller classes like BH, BU, and TTM in Table 12).

**Q9. Is code, data, or other artifacts available to enable replication?**
No code repository is mentioned anywhere in the paper. The dataset is sourced from a public Kaggle listing (ref. 30, "bct-image-dataset" by sankarlalmurugesan), but the paper's own "Data availability" statement says the datasets used in the study are available from the corresponding author "on reasonable request" rather than through a fully open public link with train/val/test splits. No mention is made of released model weights, hyperparameter configuration files, or a GitHub/Zenodo artifact, meaning replication would require re-implementing the described architecture and training setup (Tables 3 and 4) from scratch and independently sourcing/splitting the Kaggle dataset.

---

## **Title:** CapsNet-XAI: exploring the fusion of capsule networks and XAI for transparent leaf disease identification & classification

This is a paywalled article — only the abstract, references, and metadata are accessible, not the full text.

---

**1. Lacking of the Paper / Research Gap**
The paper positions itself against traditional models that struggle to capture spatial hierarchies and lack interpretability in leaf disease identification. It targets this gap by combining capsule networks (which preserve part-whole spatial relationships via dynamic routing-by-agreement) with multiple XAI techniques to make the decision-making process transparent. Beyond the abstract, I can't verify what specific limitations of prior CapsNet or XAI-leaf-disease work (e.g., cited works on Gabor capsule networks, PotCapsNet, grape/tomato capsule classifiers) they explicitly critique, since the methodology/related-work sections are behind the paywall.

**2. Used Dataset**
Four heterogeneous datasets: PlantVillage1, PlantDoc, Mango, and PlantVillage. The reference list confirms these correspond to standard public sources — a PlantVillage dataset, a PlantDoc dataset, and a Mango leaf disease dataset (refs 33–36).

**3. Repository of the Summary (Results)**
The model shows accuracy of 98.95%, 98.01%, 99.46%, and 98.76%, on respective datasets (PlantVillage1, PlantDoc, Mango, PlantVillage, in that order per the abstract). XAI methods used to explain the model: LRP, Grad-CAM, LIME, Integrated Gradients, SmoothGrad, and Guided Backpropagation. Note: no code/artifact repository is linked — the paper's own data-availability statement says no datasets were generated or analysed during the current study (i.e., they reused existing public datasets, and no new repository was released).

**4. Remarks / Comments**

- This is a **paywalled Springer Evolutionary Intelligence article** (published online 28 Feb 2026); I only had access to the abstract, figures list, and references — not the full methodology, ablation, or comparison tables. If your team needs the full text (pipeline details, statistical tests, limitations section), you'll need institutional access or the $39.95 purchase.
- Technically, it's a nice fit for your team's line of review: CapsNet + multi-XAI fusion, cross-dataset generalization across 4 heterogeneous crop datasets — directly relevant to your multi-crop XAI theme.
- One caution: capsule networks are computationally heavier than standard CNNs/ViTs due to dynamic routing — worth checking (once you get full-text access) whether they report inference time/efficiency trade-offs, since that's often a limitation left out of abstracts.

Since this article is paywalled and only the abstract/references/metadata are accessible, several answers below are limited to what's stated in the abstract — details from the full methodology/results sections aren't available to me.

**Q1. Problem & importance:** Traditional leaf disease models struggle to capture spatial hierarchies and lack interpretability, which matters for real-world agricultural disease management where trust in model decisions is critical.

**Q2. Data:** Four heterogeneous public datasets — PlantVillage1, PlantDoc, Mango (leaf disease), and PlantVillage. Source/size/timeframe/splits/collection process/ethics aren't detailed in the abstract; the paper's data-availability statement notes no new datasets were generated (existing public datasets reused).

**Q3. Features/inputs:** Not specified in accessible content — likely raw leaf images processed through capsule network layers, but feature engineering/selection details are in the full methodology (paywalled).

**Q4. Methods & pipeline:** CapsNet-XAI model — capsule network with dynamic routing-by-agreement, squash function, and vote computation to preserve part-whole spatial relationships; capsule output normalization encodes feature-presence probabilities; dimensional consistency maintained during routing for generalization. Decision-making explained via six XAI techniques: LRP, Grad-CAM, LIME, Integrated Gradients, SmoothGrad, Guided Backpropagation. Full pipeline (preprocessing, training/validation procedure) not visible.

**Q5. Baselines:** Not stated in the abstract; reference list suggests prior capsule-network variants (Gabor CapsNet, PotCapsNet, grape-leaf CapsNet) and CNN/ViT models (EfficientNet, Swin Transformer, YOLOv8) as likely comparison points, but explicit baselines used aren't confirmed without full text.

**Q6. Evaluation:** Accuracy reported per dataset; cross-dataset generalization tested to assess interpretability consistency. Statistical tests, confusion matrices, or user studies aren't mentioned in the abstract.

**Q7. Key results:** Accuracy of 98.95% (PlantVillage1), 98.01% (PlantDoc), 99.46% (Mango), and 98.76% (PlantVillage) — claimed as state-of-the-art performance. Direct numeric comparison to baselines not available from abstract alone.

**Q8. Limitations:** Not disclosed in the abstract. Capsule networks are generally known to be computationally expensive (dynamic routing overhead), which is a plausible but unconfirmed limitation here; full limitations section is behind the paywall.

**Q9. Artifact availability:** No — the data-availability statement explicitly says no datasets were generated or analysed during the study, and no code/model repository is linked.

---

## **"OptiNet-B3: a lightweight explainable deep learning model for multiclass classification of fruit and leaf diseases"** (Naveen & Ajitha, Scientific Reports, 2025).

**1. Research Gap**
Existing deep learning models for plant disease detection tend to be crop-specific (not generalizing across fruit/leaf types), rely on heavy, parameter-dense architectures unsuitable for real-time mobile/edge deployment, and lack robustness to real-world variations in image quality, orientation, and lighting. Although recent deep-learning–based approaches have demonstrated impressive accuracy in detecting individual plant diseases, they often exhibit limitations: they are tailored to a single crop and don't generalize, they rely on very deep or parameter-heavy architectures impractical for real-time deployment, and they lack robustness to variations in image quality, orientation, and lighting common in agricultural settings. This motivated a unified, lightweight, explainable model spanning multiple crops (apple, banana, orange) and both fruit and leaf images.

**2. Dataset**
Two public datasets from Kaggle were used: Fruits (D-I) contained 13,602 images across 6 classes (Fresh/Rotten Apple, Banana, Orange), and Leaves (D-II) included 11,199 images across 8 classes (Apple Scab, Apple Black Rot, Apple Cedar Rust, Apple Healthy, Banana Healthy, Banana Sigatoka, Banana Xanthomonas Wilt, Orange Huanglongbing). Data was split 70/10/20 (train/val/test), with normalization, cropping, and rotation augmentation applied.

**3. Summary of Approach/Method**
OptiNet-B3 builds on an EfficientNet-B3 backbone, enhanced with Mish activation (smoother gradient flow), CBAM (dual channel+spatial attention), Group Normalization (stable training on small batches), and Knowledge Distillation (student learns from a larger teacher model via combined cross-entropy + KL-divergence loss). It's compared against DenseNet121, ResNet50, MobileNetV3, and InceptionV3, evaluated via 5-fold cross-validation and paired t-tests, with Grad-CAM used for explainability. Results: OptiNet-B3 substantially outperformed baselines, achieving 98.12% and 99.23% accuracy on the fruit and leaf datasets, respectively, with only 12M parameters, 1.8B FLOPs, and 8.2ms inference time — making it feasible for mobile/edge deployment.

**4. Remarks/Limitations**

- Authors acknowledge: the datasets were collected under relatively controlled conditions and may not capture the full variability encountered in the field; certain rare disease classes are underrepresented; and real-world mobile/edge deployment involves additional constraints like limited processing power, energy consumption, connectivity, and farmer-facing interface needs.
- Only 3 crops (apple, banana, orange) — not truly "multi-crop" in a broad agronomic sense compared to your team's wider scope.
- Explainability is limited to Grad-CAM/Grad-CAM++ heatmap visual inspection (qualitative overlap with expert annotations), not deeper mechanistic interpretability (e.g., no LRP, no quantitative XAI metrics like localization IoU).
- Data availability is only "on reasonable request" — no public code/artifact repository link provided, limiting reproducibility for your team's artifact-availability dimension (Q9).
- Given your team's focus on cross-dataset generalization, note this paper does NOT test cross-dataset transfer — models are trained/tested within the same two curated datasets only.

**Q1. Problem addressed & importance:** Existing deep learning models for plant disease detection are typically crop-specific, rely on heavy, parameter-dense architectures unsuitable for real-time mobile/edge deployment, and lack robustness to real-world variation in image quality, orientation, and lighting — this matters because small-scale farmers need lightweight, accurate, deployable tools to enable early disease intervention, reduce yield losses, and minimize unnecessary chemical treatments.

**Q2. Data used:** Two public Kaggle datasets — Fruits (D-I): 13,602 images, 6 classes (Fresh/Rotten Apple, Banana, Orange); Leaves (D-II): 11,199 images, 8 classes (Apple Scab, Apple Black Rot, Apple Cedar Rust, Apple Healthy, Banana Healthy, Banana Sigatoka, Banana Xanthomonas Wilt, Orange Huanglongbing); split 70% train/10% val/20% test; no timeframe, licensing, or ethics/consent details are mentioned, as the images are pre-collected open-platform (mixed lab and in-field) data.

**Q3. Features/inputs:** Inputs are raw RGB fruit and leaf images; no hand-engineered features were used — features are automatically learned via the EfficientNet-B3 CNN backbone, with preprocessing limited to pixel normalization (0–1 range), cropping (to remove background and isolate the region of interest), and rotation augmentation (−30° to 30°) to improve robustness and generalization.

**Q4. Methods/models & pipeline:** The proposed OptiNet-B3 extends EfficientNet-B3 with Mish activation, CBAM (channel+spatial dual attention), Group Normalization, and Knowledge Distillation (student-teacher with combined cross-entropy + KL-divergence loss); the pipeline is preprocessing (normalize/crop/rotate) → EfficientNet-B3 + Mish feature extraction → CBAM attention refinement → GN-stabilized dual FC layers → KD-guided training → softmax classification.

**Q5. Baselines & rationale:** DenseNet121 (dense feature reuse), ResNet50 (residual connections for deep training), MobileNetV3 (lightweight, mobile-optimized), and InceptionV3 (multi-scale inception modules) were chosen as they represent standard, widely-used pre-trained ConvNet architectures spanning a range of depth, efficiency, and accuracy trade-offs relevant to agricultural image classification.

**Q6. Evaluation:** Accuracy, precision, recall, specificity, F1-score, and AUC were used across training, validation (5-fold cross-validation), and testing phases; statistical significance was confirmed via paired t-tests (p<0.05, specifically p<0.03 for all model comparisons) between OptiNet-B3 and each baseline across folds; no user studies were conducted.

**Q7. Key results:** OptiNet-B3 achieved 98.12% accuracy (D-I) and 99.23% accuracy (D-II) in testing, outperforming DenseNet121 (96.10%/97.10%), ResNet50 (95.40%/96.50%), MobileNetV3 (94.90%/95.90%), and InceptionV3 (95.70%/96.80%); it also had the best efficiency trade-off (12M parameters, 1.8B FLOPs, 8.2ms inference) versus InceptionV3's 5.7B FLOPs/14.6ms, and outperformed several SOTA models cited in discussion (Hybrid CNN 97.1%, CTPlantNet 98.28%, DenseNet-201 99.08%).

**Q8. Limitations/biases:** Datasets were collected under relatively controlled conditions and may not capture full field variability; certain rare disease classes are underrepresented; real-world mobile/edge deployment faces added constraints (processing power, energy, connectivity, farmer-facing UI); and environmental noise (overlapping leaves, weather, pest damage) common in the field isn't represented in the curated datasets, potentially limiting generalizability.

**Q9. Artifact availability:** No public code repository is provided; data is stated as "available from the corresponding author on reasonable request," limiting direct reproducibility.
