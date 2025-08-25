import re
import pytesseract
import cv2
import pandas as pd
from datetime import datetime



# Path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Extract Aadhaar details
def extract_aadhaar_details(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image. Please check the file path: {image_path}")
        return

    # OCR read
    text = pytesseract.image_to_string(img)

    # Aadhaar Number
    aadhaar_match = re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", text)
    aadhaar_number = aadhaar_match.group(0).replace(" ", "") if aadhaar_match else None

    # DOB - DD/MM/YYYY or DD-MM-YYYY
    dob_match = re.search(r"\b(\d{2})[\/\-](\d{2})[\/\-](\d{4})\b", text)
    dob = dob_match.group(0) if dob_match else None

    # Year of Birth
    yob_match = re.search(r"Year of Birth\s*[:\-]?\s*(\d{4})", text, re.IGNORECASE)
    yob = int(yob_match.group(1)) if yob_match else None

    # Gender
    gender_match = re.search(r"\b(Male|Female|Transgender)\b", text, re.IGNORECASE)
    gender = gender_match.group(0).capitalize() if gender_match else None

    # Age (if explicitly written)
    age_match = re.search(r"Age\s*[:\-]?\s*(\d+)", text, re.IGNORECASE)
    age = int(age_match.group(1)) if age_match else None

    # Age calculation from DOB or YOB if needed
    if dob:
        day, month, year = map(int, dob_match.groups())
        birth_date = datetime(year, month, day)
        today = datetime.today()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    elif yob:
        age = datetime.today().year - yob

    # --- Name Extraction ---
    name_lines = [line.strip() for line in text.split("\n") if line.strip()]
    name = None

    # If DOB found, take the line before it
    if dob and dob in name_lines:
        idx = name_lines.index(dob)
        if idx > 0:
            name = name_lines[idx - 1].strip()

    # If YOB found and no name yet
    elif yob and str(yob) in name_lines:
        idx = name_lines.index(str(yob))
        if idx > 0:
            name = name_lines[idx - 1].strip()

    # If still no name, find first "likely name" in English text
    if not name:
        for line in name_lines:
            if re.match(r"^[A-Za-z][A-Za-z\s\.]+$", line) and len(line) > 3:
                name = line.strip()
                break

    # Final Data
    data = {
        "Name": [name],
        "Aadhaar Number": [aadhaar_number],
        "DOB": [dob or (str(yob) if yob else None)],
        "Age": [age],
        "Gender": [gender]
    }

    df = pd.DataFrame(data)
    print(df)
    df.to_csv("aadhaar_data.csv", index=False)
    print("Saved to aadhaar_data.csv")

#Extract pan_card details

def extract_pan_details(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found.")
        return

    # Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # OCR
    text = pytesseract.image_to_string(gray, lang='eng')
    lines = [line.strip().upper() for line in text.split("\n") if line.strip()]

    # Remove known headers
    ignore_words = [
        "INCOME TAX DEPARTMENT", "GOVT. OF INDIA",
        "PERMANENT ACCOUNT NUMBER CARD"
    ]
    lines = [line for line in lines if line not in ignore_words]

    # PAN number
    clean_text = " ".join(lines)
    pan_match = re.search(r"\b([A-Z]{5}\d{4}[A-Z])\b", clean_text.replace(" ", ""))
    pan_number = pan_match.group(1) if pan_match else None

    # DOB
    dob_match = re.search(r"(\d{2})[\/\-](\d{2})[\/\-](\d{4})", clean_text)
    dob = dob_match.group(0) if dob_match else None
    age = None
    if dob_match:
        d, m, y = map(int, dob_match.groups())
        today = datetime.today()
        age = today.year - y - ((today.month, today.day) < (m, d))

    # Detect name & father's name
    name, father_name = None, None
    if dob and dob in lines:
        idx = lines.index(dob)
        if idx >= 2:
            name = lines[idx - 2]
            father_name = lines[idx - 1]
    else:
        # fallback: first two lines after headers filtered
        if len(lines) >= 3:
            name, father_name = lines[0], lines[1]

    # Save
    df = pd.DataFrame([{
        "Name": name,
        "PAN Number": pan_number,
        "DOB": dob,
        "Age": age,
        "Father's Name": father_name
    }])
    print(df)
    df.to_csv("pancard_data.csv", index=False)
    print("Saved to pancard_data.csv")

#Extract licence details

import re
import pytesseract
import cv2
import pandas as pd
from datetime import datetime

def extract_licence_details(image_path):
    # 1. Load and preprocess image for better OCR
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    # Try adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10
    )
    # Optionally upscale text
    scaled = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 2. OCR extraction
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(scaled, config=custom_config)
    # Print OCR text for debugging and regex improvement
    print("===== OCR OUTPUT =====\n", text)

    # 3. Use robust regex/line-based parsing for key fields
    
    # License Number
    dl_number = None
    for pat in [
        r'DL\s*No\.?\s*[:\-]?\s*([A-Z]{2}\d{2}\s?\d{11})',
        r'([A-Z]{2}\d{2}\s?\d{11})',
        r'(TN99\s*20190000999)'
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            dl_number = m.group(1).replace(" ", "")
            break

    # Name & Father/Husband Name
    name, father_name = None, None
    # Use line-wise context for these fields
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if re.match(r'Name\s*[:-]?', line, re.IGNORECASE):
            # Try extracting after the label if present
            name_part = re.sub(r'Name\s*[:-]?', '', line, flags=re.IGNORECASE).strip()
            name = name_part if len(name_part) > 2 else (lines[idx+1] if idx+1 < len(lines) else None)
        if re.match(r'Son/Daughter/Wife of\s*[:-]?', line, re.IGNORECASE):
            parent_part = re.sub(r'Son/Daughter/Wife of\s*[:-]?', '', line, flags=re.IGNORECASE).strip()
            father_name = parent_part if len(parent_part) > 2 else (lines[idx+1] if idx+1 < len(lines) else None)

    # Fallback: region-based extraction if above fails
    if not name and len(lines) >= 6:
        for idx, line in enumerate(lines):
            if "Name" in line and idx+1 < len(lines):
                possible_name = lines[idx+1].strip()
                if re.match(r"^[A-Z][A-Z\s\.]+$", possible_name):
                    name = possible_name
                    break

    if not father_name:
        for line in lines:
            if re.search(r"Son/Daughter/Wife of", line, re.IGNORECASE):
                parts = line.split("of")
                if len(parts) > 1:
                    father_name = parts[-1].strip()

    # Date of Birth (handle both formats)
    dob = None
    dob_m = re.search(r'Date of Birth\s*:?\s*(\d{2}[-/]\d{2}[-/]\d{4})', text)
    if not dob_m:
        dob_m = re.search(r'(\d{2}[-/]\d{2}[-/]\d{4})', text)
    dob = dob_m.group(1) if dob_m else None

    # Age
    age = None
    if dob:
        day, month, year = map(int, re.findall(r"\d+", dob))
        today = datetime.today()
        age = today.year - year - ((today.month, today.day) < (month, day))

    # Blood Group
    blood_group = None
    for pat in [r'Blood Group\s*:?\s*([A-Z][A-Z0-9+\-]*)', r'([ABO]{1,2}[+\-])']:
        bg_m = re.search(pat, text, re.IGNORECASE)
        if bg_m:
            blood_group = bg_m.group(1).strip()
            break

    # Date of Issue
    date_of_issue = None
    doi_m = re.search(r'Date of Issue\s*:?\s*(\d{2}[-/]\d{2}[-/]\d{4})', text)
    if not doi_m:
        doi_m = re.search(r'(\d{2}[-/]\d{2}[-/]\d{4})', text)
    date_of_issue = doi_m.group(1) if doi_m else None

    # Valid Till
    valid_till = None
    vt_m = re.search(r'Valid Till\s*:?\s*(\d{2}[-/]\d{2}[-/]\d{4})', text)
    if not vt_m:
        if date_of_issue:
            idx = text.find(date_of_issue)
            after = text[idx+10:] if idx != -1 else text
            vt = re.findall(r'(\d{2}[-/]\d{2}[-/]\d{4})', after)
            valid_till = vt[0] if vt else None
    else:
        valid_till = vt_m.group(1)

    # Prepare DataFrame
    data = {
        "DL Number": [dl_number],
        "Name": [name],
        "Father/Husband Name": [father_name],
        "Date of Birth": [dob],
        "Age": [age],
        "Blood Group": [blood_group],
        "Date of Issue": [date_of_issue],
        "Valid Till": [valid_till],
    }

    df = pd.DataFrame(data)
    print(df)
    df.to_csv("licence_data.csv", index=False)
    print("Saved to licence_data.csv")


# Example usage:
image_path = r"images\\driving_licence.jpg"  # Update path as needed
extract_licence_details(image_path)

# Example usage:
image_path = r"images\\pancard_1.jpg"
extract_pan_details(image_path)


# Example usage
image_path1 = r"images\\aadhaar.jpg"
extract_aadhaar_details(image_path1)
