## What the data actually looks like

29,191 images across **3 crops, 15 classes**: Papaya (8 classes), Potato (3 classes), Rice (4 classes). Class sizes range from Rice_Healthy Leaf (252) to Papaya_Curl (3,890) — a ~15x imbalance, so accuracy alone will be a misleading metric here; you need macro-F1 and per-class recall as first-class metrics, not afterthoughts.

**Critical issue to fix before anything else:** the Papaya images are pre-augmented — each base image has 5 variants (`image_100_aug_1` through `_aug_5`), for 11,835 unique base images inflated to a much larger file count for Papaya specifically. Potato and Rice are *not* augmented this way (1 row per base image). If you do a random train/val/test split on the raw CSV as-is, augmented copies of the *same* underlying leaf photo will leak across splits for Papaya classes — your model will partly memorize the base image rather than generalize, and your Papaya accuracy will look better than it really is. This is exactly the kind of thing your team flagged as a red flag in the papers you reviewed (the "too clean" 95–99% cross-test numbers), so it's worth getting right rather than repeating it yourselves.

**Fix:** split by base image ID (strip `_aug_N`) using `GroupShuffleSplit` or similar, so all 5 augmented copies of one base image land in the same split. For Potato/Rice, a normal stratified split is fine since there's no augmentation to leak.

## Roadmap

**1. Data pipeline (before any modeling)**
Group-aware stratified split (70/15/15 or 80/10/10), grouped by base_id and stratified by class_name so rare classes (Papaya_Mealybug: 910, Rice_Healthy Leaf: 252) still appear in val/test. Do this once, save the split as a CSV/JSON, and reuse it for every model so results are comparable.

**2. Handle imbalance**
Use class-weighted loss (inverse frequency or effective number of samples) or focal loss instead of plain cross-entropy. Avoid naive oversampling of the pre-augmented Papaya images specifically, since you'd be duplicating near-identical images — if you oversample, do it on Potato/Rice raw images, or apply fresh augmentation (not reuse of the same 5 variants) to Papaya's smaller classes like Anthracnose/BacterialSpot/Mealybug.

**3. Transfer learning — this is your supervisor's ask, and it's the right call given your data size**
Start with an ImageNet-pretrained backbone rather than training from scratch: EfficientNetB0/B3 or ResNet50 are good first choices given the moderate dataset size (~29K images, 15 classes) — heavier hybrids like the ResViT-152 you reviewed need far more data/compute to justify their cost. Two-phase fine-tuning: (a) freeze the backbone, train only a new classification head for a few epochs to let it adapt to your 15-class problem; (b) unfreeze the last 20–30% of layers and fine-tune end-to-end at a low learning rate (discriminative LR: smaller for early layers, larger for later layers). Use standard ImageNet normalization stats and resize to whatever your backbone expects (224×224 for EfficientNetB0/ResNet50).

**4. Training protocol**
AdamW with cosine or step LR decay, label smoothing, early stopping on macro-F1 (not accuracy) with patience, and 5-fold stratified group cross-validation if compute allows — this is the pattern nearly every paper in your review used, and it'll make your results comparable to theirs.

**5. Evaluation**
Report accuracy, macro-F1, per-class precision/recall (especially for the small classes), and a confusion matrix — this will directly show whether Rice_Healthy Leaf or Papaya_Mealybug (your smallest classes) are being confused with visually similar diseases. If you want a cross-dataset generalization angle consistent with your team's stated priority, consider testing your trained model against a public dataset with overlapping crops (e.g., a Rice or Potato subset of PlantVillage) — even a small out-of-distribution test adds credibility your reviewed papers often lacked.

**6. XAI layer**
Once accuracy is solid, add Grad-CAM or Grad-CAM++ on the fine-tuned backbone for qualitative checks, and if time permits, a lightweight quantitative check (e.g., do heatmaps concentrate on leaf lesion regions vs. background) — this was the most common weak point ("qualitative only, no IoU validation") across the papers you catalogued, so even a small quantitative pass would put your work ahead of most of them.

**7. Only after a solid single-backbone baseline**
Consider a hybrid CNN+attention (CBAM) or CNN+Transformer setup like the papers you reviewed, but only if your baseline transfer-learning model is already stable and well-evaluated — added architectural complexity without a clean data/eval pipeline underneath it is exactly the failure mode that made a couple of those papers hard to trust.

Want me to sketch the actual training code (e.g., a PyTorch or TensorFlow pipeline with the grouped split baked in) as a starting point?