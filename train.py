import cv2
import numpy as np
from ultralytics import YOLO
import os

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = r"C:\Users\ASUS!\Desktop\seg_train\runs\segment\runs\segmentation\crop_weed_seg\weights\best.pt"
IMAGE_PATH = r"C:\Users\ASUS!\Desktop\seg_train\test2.jpg"

# Output folders
BASE_OUTPUT_DIR = "outputs"
CROP_DIR = os.path.join(BASE_OUTPUT_DIR, "crop_mask")
WEED_DIR = os.path.join(BASE_OUTPUT_DIR, "weed_instances_vis")
HIGHLIGHT_DIR = os.path.join(BASE_OUTPUT_DIR, "highlighted")

os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(WEED_DIR, exist_ok=True)
os.makedirs(HIGHLIGHT_DIR, exist_ok=True)

# =====================================================
# LOAD MODEL & IMAGE
# =====================================================
model = YOLO(MODEL_PATH)

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError("❌ Image not found")

orig = img.copy()
h, w = img.shape[:2]

# =====================================================
# YOLO → CROP MASK
# =====================================================
results = model(img, device="cpu")[0]

crop_mask = np.zeros((h, w), dtype=np.uint8)

if results.masks is not None:
    for i, cls in enumerate(results.boxes.cls):
        if int(cls) == 0:  # crop
            mask = results.masks.data[i].cpu().numpy()
            mask = cv2.resize(mask, (w, h))
            crop_mask = np.maximum(crop_mask, mask)

crop_mask = (crop_mask * 255).astype(np.uint8)

# =====================================================
# REMOVE CROP REGION
# =====================================================
no_crop = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(crop_mask))

# =====================================================
# WEED PIXEL DETECTION (HSV + ExG)
# =====================================================
hsv = cv2.cvtColor(no_crop, cv2.COLOR_BGR2HSV)

lower_green = np.array([30, 30, 30])
upper_green = np.array([90, 255, 255])
hsv_mask = cv2.inRange(hsv, lower_green, upper_green)

b, g, r = cv2.split(no_crop)
exg = 2 * g.astype(np.int16) - r.astype(np.int16) - b.astype(np.int16)
exg = np.clip(exg, 0, 255).astype(np.uint8)
_, exg_mask = cv2.threshold(exg, 25, 255, cv2.THRESH_BINARY)

weed_mask = cv2.bitwise_or(hsv_mask, exg_mask)

# Remove low-saturation soil noise
_, s, _ = cv2.split(hsv)
weed_mask[s < 40] = 0

# =====================================================
# INSTANCE SEPARATION (UNCHANGED)
# =====================================================
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_OPEN, kernel)

num_labels, labels = cv2.connectedComponents(weed_mask)

final_vis = orig.copy()
weed_instances = []

for label_id in range(1, num_labels):
    instance_mask = np.zeros((h, w), dtype=np.uint8)
    instance_mask[labels == label_id] = 255

    area = cv2.countNonZero(instance_mask)
    if area < 120:
        continue

    contours, _ = cv2.findContours(
        instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        continue

    cnt = contours[0]

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    weed_id = f"weed_{len(weed_instances) + 1}"

    weed_instances.append({
        "id": weed_id,
        "mask": instance_mask,
        "centroid": (cx, cy)
    })

    cv2.drawContours(final_vis, [cnt], -1, (0, 0, 255), 2)
    cv2.circle(final_vis, (cx, cy), 4, (255, 0, 0), -1)
    cv2.putText(
        final_vis,
        weed_id,
        (cx + 5, cy - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 0, 0),
        1
    )

# =====================================================
# HIGHLIGHT IMAGE (UNCHANGED)
# =====================================================
highlight = orig.copy()
highlight[crop_mask > 0] = (0, 255, 0)

for w_inst in weed_instances:
    highlight[w_inst["mask"] > 0] = (0, 0, 255)

highlight = cv2.addWeighted(orig, 0.4, highlight, 0.6, 0)

# =====================================================
# SAVE OUTPUTS (NEW PART)
# =====================================================
cv2.imwrite(os.path.join(CROP_DIR, "crop_mask.png"), crop_mask)
cv2.imwrite(os.path.join(WEED_DIR, "weed_instances.png"), final_vis)
cv2.imwrite(os.path.join(HIGHLIGHT_DIR, "highlighted_crops_weeds.png"), highlight)

print("✅ Outputs saved:")
print(f"- Crop mask → {CROP_DIR}")
print(f"- Weed instances → {WEED_DIR}")
print(f"- Highlighted image → {HIGHLIGHT_DIR}")
print(f"Total weed instances detected: {len(weed_instances)}")

# =====================================================
# DISPLAY (OPTIONAL – unchanged)
# =====================================================
cv2.imshow("Crop Mask", crop_mask)
cv2.imshow("Weed Instance Segmentation (Laser Ready)", final_vis)
cv2.imshow("Highlighted Crops & Weeds", highlight)

cv2.waitKey(0)
cv2.destroyAllWindows()
