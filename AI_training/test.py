import os
import cv2
import numpy as np
import argparse

def draw_bounding_boxes(image_folder, label_folder, output_folder=None, show_images=True, use_rotated_box=True):
    """
    Draw bounding boxes on images based on segmentation data from label files.
    
    Args:
        image_folder (str): Path to folder containing images
        label_folder (str): Path to folder containing segmentation labels (txt format)
        output_folder (str, optional): If provided, save annotated images to this folder
        show_images (bool): Whether to display images or not
        use_rotated_box (bool): Use rotated bounding box for better fit with segmentation
    """
    # Create output folder if needed
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(image_folder) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    print(f"Found {len(image_files)} images in {image_folder}")
    
    processed_count = 0
    for img_name in image_files:
        img_path = os.path.join(image_folder, img_name)
        # Assume label file has the same name but .txt extension
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(label_folder, label_name)
        
        if not os.path.exists(label_path):
            print(f"No label file for {img_name}, skipping")
            continue
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read image: {img_path}, skipping")
            continue
            
        # Get image dimensions for handling normalized coordinates
        img_height, img_width = img.shape[:2]
        
        # Create a copy for drawing
        img_with_boxes = img.copy()
        
        try:
            # Read segmentation points from label
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line_idx, line in enumerate(lines):
                try:
                    # Parse coordinates from line
                    points = list(map(float, line.strip().split()))
                    
                    # Check if points are normalized (between 0 and 1)
                    normalized = all(0 <= p <= 1 for p in points)
                    
                    # Create array of points
                    segmentation_points = []
                    for i in range(0, len(points), 2):
                        if i+1 < len(points):  # Ensure we have both x and y
                            x, y = points[i], points[i+1]
                            
                            # Convert normalized coordinates if needed
                            if normalized:
                                x, y = x * img_width, y * img_height
                                
                            segmentation_points.append((x, y))
                    
                    if not segmentation_points:
                        continue
                        
                    # Convert to numpy array, keeping float precision
                    points_array = np.array(segmentation_points, dtype=np.float32).reshape(-1, 2)
                    
                    # Draw the segmentation polygon with a unique color for each object
                    color_poly = (0, 165, 255)  # Orange for polygon
                    points_int = np.round(points_array).astype(np.int32)
                    cv2.polylines(img_with_boxes, [points_int], True, color_poly, 2)
                    
                    if use_rotated_box:
                        # Find rotated bounding box (minimum area rectangle)
                        rect = cv2.minAreaRect(points_array)
                        box_points = cv2.boxPoints(rect)
                        box_points_int = np.intp(box_points)
                        
                        # Draw rotated bounding box
                        color_box = (0, 255, 0)  # Green for bounding box
                        cv2.drawContours(img_with_boxes, [box_points_int], 0, color_box, 2)
                        
                        # Get center, width, height, and angle of rotated box
                        (center_x, center_y), (width, height), angle = rect
                        
                        # Add text with dimensions and angle
                        text = f"Object {line_idx+1}: {width:.1f}x{height:.1f}, {angle:.1f}°"
                        cv2.putText(img_with_boxes, text, 
                                   (int(center_x), int(center_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, color_box, 1)
                    else:
                        # Find standard bounding box
                        x, y, w, h = cv2.boundingRect(points_array)
                        
                        # Draw standard bounding box
                        color_box = (0, 255, 0)  # Green for bounding box
                        cv2.rectangle(img_with_boxes, (int(x), int(y)), 
                                     (int(x + w), int(y + h)), color_box, 2)
                        
                        # Add text with coordinates and dimensions
                        text = f"Object {line_idx+1}: ({int(x)},{int(y)}) {w:.1f}x{h:.1f}"
                        cv2.putText(img_with_boxes, text, 
                                   (int(x), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, color_box, 1)
                     
                except Exception as e:
                    print(f"Error processing line in {label_name}: {e}")
                    continue
            
            processed_count += 1
            
            # Save the annotated image if requested
            if output_folder:
                output_path = os.path.join(output_folder, img_name)
                cv2.imwrite(output_path, img_with_boxes)
                print(f"Saved annotated image to {output_path}")
            
            # Show image if requested
            if show_images:
                cv2.imshow(f"Bounding Boxes - {img_name}", img_with_boxes)
                print(f"Processed {img_name} - Press any key to continue or 'q' to quit")
                key = cv2.waitKey(0)
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
        
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            continue
    
    if show_images:
        cv2.destroyAllWindows()
    
    print(f"Processed {processed_count} out of {len(image_files)} images")

def main():
    parser = argparse.ArgumentParser(description="Draw bounding boxes from segmentation data")
    parser.add_argument("--image_dir", type=str, 
                        default="/home/khoa_is_sleep/screws/DATA/Data_screws(origin)/images",
                        help="Directory containing images")
    parser.add_argument("--label_dir", type=str, 
                        default="/home/khoa_is_sleep/screws/DATA/Data_screws(origin)/labels",
                        help="Directory containing segmentation label files")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save annotated images (optional)")
    parser.add_argument("--show", action="store_true", default=True,
                       help="Show images with bounding boxes")
    parser.add_argument("--rotated", action="store_true", default=True,
                       help="Use rotated bounding boxes for better fit (default: True)")
    
    args = parser.parse_args()
    
    draw_bounding_boxes(
        args.image_dir,
        args.label_dir,
        output_folder=args.output_dir,
        show_images=args.show,
        use_rotated_box=args.rotated
    )

if __name__ == "__main__":
    main()
