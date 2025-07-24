import cv2
import numpy as np
import os
import pytesseract
from pytesseract import Output

def extract_signature(image_path, output_folder='extracted_signatures'):
    """
    Extract signature from either:
    1. Plain white page (just signature)
    2. Complex document (forms/cheques with text)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if image is plain white background with just signature
    if is_plain_white_page(img):
        print("Detected plain white page with signature")
        signature = extract_from_plain_page(img)
    else:
        print("Detected complex document")
        signature = extract_from_complex_document(img, image_path,output_folder)
    
    return signature

def is_plain_white_page(img, threshold=0.95):
    """Check if image is mostly white background"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    white_ratio = cv2.countNonZero(thresh) / gray.size
    return white_ratio > threshold

def extract_from_plain_page(img):
    """Extract signature from plain white page"""
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Get largest contour (signature)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add padding
    padding = 20
    x, y = max(0, x-padding), max(0, y-padding)
    w, h = min(img.shape[1]-x, w+2*padding), min(img.shape[0]-y, h+2*padding)
    
    return img[y:y+h, x:x+w]

def extract_from_complex_document(img, image_path,output_folder):
    """Extract signature from complex document (forms/cheques)"""
    # Try to find signature using text labels
    signature_count = find_signatures_near_labels(img, image_path,output_folder)
    
    if signature_count > 0:
        print(f"Found {signature_count} signatures using label detection")
        return True
    
    # Fall back to common areas if no signatures found via labels
    print("No signatures found via labels, trying common areas...")
    return extract_from_common_areas(img, image_path,output_folder)

def find_signatures_near_labels(img,image_path, output_folder):
    """Find signatures by detecting signature labels in the document"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, output_type=Output.DICT)
    
    # Target labels that might indicate signature locations
    target_labels = [
        "signature", "sign", "signed", "authorized", "authorised",
        "endorsement", "endorse", "signhere", "signatureline",
        "your signature", "client signature", "customer signature"
    ]
    
    # Find all matching labels in the document
    found_labels = []
    for i, text in enumerate(data['text']):
        text_lower = text.lower().strip()
        if any(label in text_lower for label in target_labels):
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            found_labels.append({
                'text': text,
                'position': (x, y, w, h)
            })
    
    if not found_labels:
        print("No signature labels found in document")
        return 0
    
    print(f"Found {len(found_labels)} potential signature labels")
    
    # Process each found label
    signature_count = 0
    for label_info in found_labels:
        x, y, w, h = label_info['position']
        label_text = label_info['text']
        
        # Define signature area above the label
        signature_height = h * 15  # Look 15 times the label height above
        signature_width = w * 15   # Wider area to capture full signature
        signature_x = max(0, x - (signature_width - w) // 2)
        signature_y = max(0, y - signature_height)
        
        # Extract the signature region
        signature_region = img[signature_y:y, signature_x:x + signature_width]
        
        if signature_region.size == 0:
            print(f"No area above label '{label_text}' to extract signature")
            continue
        
        # Convert to grayscale and threshold to detect ink
        gray_signature = cv2.cvtColor(signature_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_signature, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Light morphological cleaning to preserve gaps
        kernel = np.ones((2, 2), np.uint8)
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean, connectivity=8)
        
        # Find the largest component (likely signature)
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0:
            print(f"No signature components found above '{label_text}'")
            continue
        
        largest_index = 1 + np.argmax(areas)
        lx, ly, lw, lh, _ = stats[largest_index]
        
        # Create mask keeping nearby components to the largest one
        mask = np.zeros_like(clean)
        for i in range(1, num_labels):
            x_comp, y_comp, w_comp, h_comp, area = stats[i]
            
            # Ignore tiny specks
            if area < 50:
                continue
            
            # Skip printed text-like components
            aspect_ratio = w_comp / h_comp if h_comp > 0 else 0
            if (aspect_ratio > 6 and h_comp < 30) or h_comp < 15:
                continue
            
            # Check if component is close to main signature
            if (x_comp < lx + lw + 20 and x_comp + w_comp > lx - 20 and
                y_comp < ly + lh + 10 and y_comp + h_comp > ly - 10):
                mask[labels == i] = 255
        
        # Get bounding box of the combined signature components
        coords = cv2.findNonZero(mask)
        if coords is None:
            print(f"No valid signature region detected above '{label_text}'")
            continue
        
        x_sig, y_sig, w_sig, h_sig = cv2.boundingRect(coords)

        # Add some padding around the signature
        padding = 25
        x_sig = max(0, x_sig - padding)
        y_sig = max(0, y_sig - padding)
        w_sig = min(signature_region.shape[1] - x_sig, w_sig + 2*padding)
        h_sig = min(signature_region.shape[0] - y_sig, h_sig + 2*padding)
        
        # Crop to the exact signature area
        exact_signature = signature_region[y_sig:y_sig+h_sig, x_sig:x_sig+w_sig]
        
        # Check if we actually found a signature (not just noise)
        if exact_signature.size > 100:  # At least 100 pixels
            output_path = os.path.join(output_folder, f"signature_{signature_count}_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, exact_signature)
            print(f"Extracted signature above '{label_text}'. Saved to {output_path}")
            signature_count += 1
        else:
            print(f"Potential noise detected above '{label_text}', ignoring")
    
    return signature_count

def extract_from_common_areas(img, image_path,output_folder):
    """Check common signature locations (e.g., bottom-right for cheques)"""
    h, w = img.shape[:2]
    
    # Define search regions (y1, y2, x1, x2, region_name)
    search_regions = [
        (int(h*0.5), h, int(w*0.4), w, "Bottom-Right"),
        (int(h*0.6), h, int(w*0.2), int(w*0.8), "Bottom-Center"),
        (0, h, int(w*0.7), w, "Right-Strip"),
        (int(h*0.5), h, 0, w, "Bottom-Half")
    ]
    
    signature_count = 0
    for y1, y2, x1, x2, region_name in search_regions:
        print(f"Checking {region_name} region...")
        roi = img[y1:y2, x1:x2]
        
        # Process the ROI using the same signature extraction logic
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((2, 2), np.uint8)
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        # Filter contours based on signature-like properties
        valid_contours = []
        min_area = max(roi.shape[0] * roi.shape[1] * 0.005, 100)
        
        for cnt in contours:
            x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
            aspect_ratio = w_cnt / float(h_cnt)
            area = cv2.contourArea(cnt)
            
            if (area > min_area and 0.2 < aspect_ratio < 5 and 
                h_cnt > roi.shape[0] * 0.05 and w_cnt > roi.shape[1] * 0.05):
                valid_contours.append(cnt)
        
        if not valid_contours:
            continue
        
        # Create mask from valid contours
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, valid_contours, -1, 255, -1)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(mask)
        padding = int(max(roi.shape) * 0.05)
        x, y = max(0, x-padding), max(0, y-padding)
        w, h = min(roi.shape[1]-x, w+2*padding), min(roi.shape[0]-y, h+2*padding)
        
        # Extract and save signature
        signature = roi[y:y+h, x:x+w]
        if signature.size > 100:
            output_path = os.path.join(output_folder, f"common_area_{signature_count}_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, signature)
            print(f"Found signature in {region_name} region. Saved to {output_path}")
            signature_count += 1
    
    return signature_count > 0

if __name__ == "__main__":
    # Example usage
    input_image = "inputs/Raw-scanned-image-of-a-sample-cheque.png"  # Replace with your image path
    extracted = extract_signature(input_image)
    
    if extracted:
        print("Signature extraction completed successfully")
    else:
        print("No signature found in the image")