import cv2
import numpy as np
import os
import string

# Path to your 13x12 image
image_path = "./image.png"
# Output directory for extracted letters
output_base = "./extracted_letters"

# Create subfolders for the 6 font variations
for i in range(6):
    os.makedirs(f"{output_base}/font_{i+1}", exist_ok=True)

# Read the full image
image = cv2.imread(image_path)
height, width, _ = image.shape

# We have 13 rows (each row has 2 letters repeated 6 times each).
num_rows = 13

# For A-Z only:
characters = list(string.ascii_uppercase)  # 26 letters => [A, B, C, ..., Z]

# Compute approximate row height
row_height = height // num_rows

# Loop through each of the 13 rows
for row in range(num_rows):
    # Determine which two letters belong in this row
    letter_index1 = row * 2
    letter_index2 = letter_index1 + 1
    if letter_index2 >= len(characters):
        break  # we've reached beyond Z

    letter1 = characters[letter_index1]  # e.g., row 0 => 'A'
    letter2 = characters[letter_index2]  # e.g., row 0 => 'B'

    # Slice out this entire row
    y1 = row * row_height
    y2 = (row + 1) * row_height if row < num_rows - 1 else height  # last row may be slightly bigger
    row_crop = image[y1:y2, 0:width]

    # Convert row_crop to grayscale and threshold to find letter contours
    gray = cv2.cvtColor(row_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by x-coordinate (left to right)
    bounding_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Filter out noise or extremely small boxes
        if w > 5 and h > 5:
            bounding_boxes.append((x, y, w, h))
    # We expect ~12 bounding boxes
    bounding_boxes.sort(key=lambda b: b[0])

    # If there are fewer than 12 found, we might not have all letters
    if len(bounding_boxes) < 12:
        print(f"Row {row}: Found {len(bounding_boxes)} letters, expected 12. Some letters may be missing or merged.")
    # If more than 12, keep only the leftmost 12
    bounding_boxes = bounding_boxes[:12]

    # For each of these 12 boxes, assign them to letter1 or letter2
    # columns 0..5 => letter1, columns 6..11 => letter2
    for col, (x, y, w, h) in enumerate(bounding_boxes):
        # Decide which letter
        letter_name = letter1 if col < 6 else letter2
        # Decide which font folder
        font_index = col % 6  # 0..5 => font_1..font_6

        # Crop out the letter with a small margin
        margin = 2
        x1 = max(0, x - margin)
        y1_local = max(0, y - margin)
        x2 = min(row_crop.shape[1], x + w + margin)
        y2_local = min(row_crop.shape[0], y + h + margin)

        # letterCrop is from row_crop coordinates
        letter_crop = row_crop[y1_local:y2_local, x1:x2]

        # Save the letter
        letter_path = os.path.join(output_base, f"font_{font_index+1}", f"{letter_name}.png")
        cv2.imwrite(letter_path, letter_crop, [cv2.IMWRITE_PNG_COMPRESSION, 0])

print("✅ Letters extracted with minimal bounding boxes and saved in high quality!")
