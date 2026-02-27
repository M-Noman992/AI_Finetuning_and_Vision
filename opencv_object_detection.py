import cv2
import os

# Check if the cascades are loaded
cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')
license_plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

if cat_cascade.empty():
    print("Error loading cat cascade.")
if license_plate_cascade.empty():
    print("Error loading license plate cascade.")

def detect_and_save(image_paths, cascade, output_dir, task_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
        )
        
        # Check if detections are found
        if len(detections) == 0:
            print(f"No detections found in {img_path}")
        else:
            for (x, y, w, h) in detections:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        output_path = os.path.join(output_dir, f"{task_name}_output{i + 1}.jpg")
        cv2.imwrite(output_path, img)
        print(f"Processed and saved: {output_path}")

cat_images = ['cat1.jpg', 'cat2.jpg']
license_plate_images = ['plate1.jpg', 'plate2.jpg']

detect_and_save(cat_images, cat_cascade, 'output_cats', 'cat')
detect_and_save(license_plate_images, license_plate_cascade, 'output_plates', 'plate')

print("Processing complete.")
