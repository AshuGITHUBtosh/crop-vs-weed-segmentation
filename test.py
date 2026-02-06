import cv2
import numpy as np
from ultralytics import YOLO

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = r"C:\Users\ASUS!\Desktop\seg_train\runs\segment\runs\segmentation\crop_weed_seg\weights\best.pt"
IMAGE_PATH = r"C:\Users\ASUS!\Desktop\seg_train\test4.jpg"

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
# YOLO INFERENCE → CROP MASK
# =====================================================
results = model(img, device="cpu")[0]

crop_mask = np.zeros((h, w), dtype=np.uint8)

if results.masks is not None:
    for i, cls in enumerate(results.boxes.cls):
        if int(cls) == 0:  # crop class
            mask = results.masks.data[i].cpu().numpy()
            mask = cv2.resize(mask, (w, h))
            crop_mask = np.maximum(crop_mask, mask)

crop_mask = (crop_mask * 255).astype(np.uint8)

# =====================================================
# REMOVE CROP REGION
# =====================================================
no_crop = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(crop_mask))

# =====================================================
# 🌿 IMPROVED WEED DETECTION
# =====================================================

# ---------- HSV GREEN DETECTION ----------
hsv = cv2.cvtColor(no_crop, cv2.COLOR_BGR2HSV)

lower_green = np.array([30, 30, 30])
upper_green = np.array([90, 255, 255])

hsv_mask = cv2.inRange(hsv, lower_green, upper_green)

# ---------- EXCESS GREEN INDEX (ExG) ----------
b, g, r = cv2.split(no_crop)
exg = 2 * g.astype(np.int16) - r.astype(np.int16) - b.astype(np.int16)
exg = np.clip(exg, 0, 255).astype(np.uint8)

_, exg_mask = cv2.threshold(exg, 25, 255, cv2.THRESH_BINARY)

# ---------- COMBINE HSV + ExG ----------
weed_mask = cv2.bitwise_or(hsv_mask, exg_mask)

# =====================================================
# MORPHOLOGY (PRESERVE THIN WEEDS)
# =====================================================
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_CLOSE, kernel)
weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_OPEN, kernel)

# =====================================================
# CONTOUR DETECTION (GROUP WEEDS)
# =====================================================
final_img = orig.copy()

contours, _ = cv2.findContours(
    weed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

for cnt in contours:
    area = cv2.contourArea(cnt)
    x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)

    # Smart filtering: keep grass-like weeds
    if area > 300 and h_cnt > 15:
        cv2.drawContours(final_img, [cnt], -1, (0, 0, 255), 2)
        cv2.putText(
            final_img,
            "Weed",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

# =====================================================
# 🌈 HIGHLIGHT CROPS & WEEDS
# =====================================================
highlight = orig.copy()

# Crops → GREEN
highlight[crop_mask > 0] = (0, 255, 0)

# Weeds → RED
highlight[weed_mask > 0] = (0, 0, 255)

# Transparency blend
alpha = 0.6
highlight = cv2.addWeighted(orig, 1 - alpha, highlight, alpha, 0)

# =====================================================
# DISPLAY RESULTS
# =====================================================
cv2.imshow("Crop Mask", crop_mask)
cv2.imshow("Weed Mask (HSV + ExG)", weed_mask)
cv2.imshow("Final Output (Contours)", final_img)
cv2.imshow("Highlighted Crops & Weeds", highlight)

cv2.waitKey(0)
cv2.destroyAllWindows()
