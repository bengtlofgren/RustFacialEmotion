import cv2
import sys
import os

def crop_face(input_path, output_paths):
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read image
    frame_bgr = cv2.imread(input_path)
    if frame_bgr is None:
        print(f"Could not read image: {input_path}")
        return False
        
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        print("No faces detected!")
        return False
    
    # Check if we have enough output paths
    if len(faces) > len(output_paths):
        print(f"Warning: Found {len(faces)} faces but only {len(output_paths)} output paths provided.")
        print("Only the first", len(output_paths), "faces will be saved.")
        faces = faces[:len(output_paths)]
    
    # Process each face
    for (x, y, w, h), output_path in zip(faces, output_paths):
        # Add some margin (1%)
        margin = 0.01
        dh = int(h * margin)
        dw = int(w * margin)
        
        # Get new coordinates with margin
        y1 = max(y - dh, 0)
        y2 = min(y + h + dh, frame_bgr.shape[0])
        x1 = max(x - dw, 0)
        x2 = min(x + w + dw, frame_bgr.shape[1])
        
        face_img_with_margin = frame_bgr[y1:y2, x1:x2]
        
        # Save cropped face
        cv2.imwrite(output_path, face_img_with_margin)
        print(f"Saved face to: {output_path}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python crop_faces.py input_image output_image1 [output_image2 output_image3 ...]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_paths = sys.argv[2:]
    
    crop_face(input_path, output_paths)