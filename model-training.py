from ultralytics import YOLO

DATA_DIR = r"C:\Users\uaeor\Downloads\Duncan\dataset_tidy"

def main():
    model = YOLO("yolov8n-cls.pt") 
    
    model.train(
        data=DATA_DIR,
        epochs=5,           # Lowered from 15 to 5 (It learns fast!)
        imgsz=224,          # [!!!] Lowered from 1024 to 224
        batch=16,           # Increased batch size (since images are smaller)
        name='yolo_tidy_classifier_fast' 
    )

if __name__ == "__main__":
    main()