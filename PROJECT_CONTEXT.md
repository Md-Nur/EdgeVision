Project: multi-crop leaf disease classification (Potato, Papaya, Rice), 12 classes (Papaya 4, Potato 4, Rice 4), deep learning + XAI.
Rules:

- Papaya images are pre-augmented (_aug_N in filenames). Never let variants of the same base image cross train/val/test. Use GroupShuffleSplit keyed on base image ID.
- Metric is macro-F1 plus per-class F1. Never report accuracy alone.
- Strong class imbalance (~15x). Use weighted or focal loss.
- Fixed seed 42. Save all splits as CSV files and commit them.
- Start with a single-backbone baseline (EfficientNet-B0). No hybrid or ViT yet.
- Never invent results. Run the code and paste real outputs.
- Work one stage at a time and stop for my review after each.
