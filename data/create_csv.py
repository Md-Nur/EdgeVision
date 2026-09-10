import os
import csv

# ==============================
# SETTINGS
# ==============================

# Main dataset folder
DATASET_DIR = "MyDataset"

# Output CSV file
OUTPUT_CSV = "dataset.csv"

# Supported image formats
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ==============================
# FIND ALL IMAGES
# ==============================

data = []

# Create label mapping
label_map = {}
next_label = 0

for crop in sorted(os.listdir(DATASET_DIR)):

    crop_path = os.path.join(DATASET_DIR, crop)

    # Skip files
    if not os.path.isdir(crop_path):
        continue

    for disease in sorted(os.listdir(crop_path)):

        disease_path = os.path.join(crop_path, disease)

        # Skip files
        if not os.path.isdir(disease_path):
            continue

        # Create unique label for each disease
        class_name = f"{crop}_{disease}"

        if class_name not in label_map:
            label_map[class_name] = next_label
            next_label += 1

        label = label_map[class_name]

        # Find images
        for filename in sorted(os.listdir(disease_path)):

            if filename.lower().endswith(IMAGE_EXTENSIONS):

                image_path = os.path.join(
                    DATASET_DIR,
                    crop,
                    disease,
                    filename
                )

                # Use forward slash
                image_path = image_path.replace("\\", "/")

                data.append([
                    image_path,
                    filename,
                    crop,
                    disease,
                    class_name,
                    label
                ])


# ==============================
# WRITE CSV
# ==============================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    # Header
    writer.writerow([
        "image_path",
        "filename",
        "crop",
        "disease",
        "class_name",
        "label"
    ])

    # Data
    writer.writerows(data)


# ==============================
# PRINT RESULT
# ==============================

print("\n===================================")
print("CSV CREATED SUCCESSFULLY!")
print("===================================")

print(f"\nTotal images: {len(data)}")
print(f"Total classes: {len(label_map)}")

print("\nClass Labels:")

for class_name, label in label_map.items():
    print(f"{label}: {class_name}")

print(f"\nCSV file: {OUTPUT_CSV}")