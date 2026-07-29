import os
from PIL import Image

# ======================================================
# CONFIGURATION
# ======================================================

INPUT_FOLDER = r"D:\ISL_1_PROJECT\isl_recognition\datasets"
OUTPUT_FOLDER = r"D:\ISL_1_PROJECT\isl_recognition\datasets_duplicated"

# Crop percentage (5%)
CROP_PERCENT = 0.05

SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp")

# ======================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def save_crop(img, crop_box, save_path):
    cropped = img.crop(crop_box)
    cropped.save(save_path)


total_images = 0

for root, dirs, files in os.walk(INPUT_FOLDER):

    relative_path = os.path.relpath(root, INPUT_FOLDER)
    output_dir = os.path.join(OUTPUT_FOLDER, relative_path)

    os.makedirs(output_dir, exist_ok=True)

    for file in files:

        if not file.lower().endswith(SUPPORTED_FORMATS):
            continue

        img_path = os.path.join(root, file)

        img = Image.open(img_path)

        width, height = img.size

        dx = int(width * CROP_PERCENT)
        dy = int(height * CROP_PERCENT)

        name, ext = os.path.splitext(file)

        # ------------------------------------------------
        # Save Original
        # ------------------------------------------------
        img.save(os.path.join(output_dir, f"{name}_original{ext}"))

        # ------------------------------------------------
        # LEFT CROP
        # Removes 5% from right
        # ------------------------------------------------
        save_crop(
            img,
            (0, 0, width - dx, height),
            os.path.join(output_dir, f"{name}_left{ext}")
        )

        # ------------------------------------------------
        # RIGHT CROP
        # Removes 5% from left
        # ------------------------------------------------
        save_crop(
            img,
            (dx, 0, width, height),
            os.path.join(output_dir, f"{name}_right{ext}")
        )

        # ------------------------------------------------
        # TOP CROP
        # Removes 5% from bottom
        # ------------------------------------------------
        save_crop(
            img,
            (0, 0, width, height - dy),
            os.path.join(output_dir, f"{name}_top{ext}")
        )

        # ------------------------------------------------
        # BOTTOM CROP
        # Removes 5% from top
        # ------------------------------------------------
        save_crop(
            img,
            (0, dy, width, height),
            os.path.join(output_dir, f"{name}_bottom{ext}")
        )

        total_images += 1

print("=" * 50)
print("Augmentation Completed!")
print(f"Original Images Processed : {total_images}")
print(f"New Images Created        : {total_images * 5}")
print(f"Saved To                 : {OUTPUT_FOLDER}")
print("=" * 50)