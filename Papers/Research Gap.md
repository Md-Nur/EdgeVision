## Research Gaps: Edge Vision

> Literature Review Sheet: [Google Sheet](https://docs.google.com/spreadsheets/d/1-b532VXTwc4OAJGN1bVVTL2XEfH2yTgqhsqjxMZQ7F8/edit?usp=sharing)

1. **Cross-dataset validation:** Most studies train and test on the same dataset. Testing on an independent dataset for rice, potato, and papaya could show how well the models actually generalize.

2. **Data leakage:** Some studies augment data before splitting, which can cause leakage and overly high accuracy. Comparing normal splits with group-aware, leakage-free splits would be a useful contribution.

3. **Quantitative XAI:** Most papers only show heatmaps. Using Grad-CAM++, LIME, SHAP, etc. with metrics like IoU, Dice, and faithfulness would make the explanations more reliable.

4. **One model for multiple crops:** Most studies build separate models for each crop. Comparing one multi-crop model with separate crop-specific models for rice, potato, and papaya is still largely unexplored.

5. **Class imbalance:** Many papers mainly report accuracy despite imbalanced classes. Comparing weighted loss, focal loss, and resampling using macro-F1 and per-class results would address this gap.

6. **Lab vs. field images:** Most models work well on controlled images but are not tested properly in real field conditions. Testing with different lighting, backgrounds, blur, and clutter would show real-world robustness.

7. **Accuracy vs. efficiency vs. explainability:** A direct comparison of ResNet, EfficientNet, ViT, and CapsNet using the same leakage-free setup, while measuring accuracy, macro-F1, parameters, speed, and XAI quality, would be valuable.

8. **Reproducibility:** Very few studies provide complete code, splits, and trained models. Sharing these with a well-designed leakage-free split could make the research more reproducible and useful.
