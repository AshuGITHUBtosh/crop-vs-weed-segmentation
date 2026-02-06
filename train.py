from ultralytics import YOLO

DATA_YAML = r"C:\Users\ASUS!\Desktop\seg_train\data.yaml"
OUTPUT_DIR = "runs/segmentation"

model = YOLO("yolov8n-seg.pt")

model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=8,
    device="cpu",   # ✅ FIXED
    project=OUTPUT_DIR,
    name="crop_weed_seg",
    patience=10,
    save=True,
    verbose=True
)

print("✅ Training complete")
