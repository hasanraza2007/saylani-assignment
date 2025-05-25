import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import easyocr
import pandas as pd
import google.generativeai as genai
import re
from dotenv import load_dotenv
import os
from pdf2image import convert_from_bytes
import io

# --- Streamlit UI Configuration (MUST BE THE FIRST ST. COMMAND) ---
# This line sets up the page layout and title, and it must come before any other st. calls.
st.set_page_config(layout="wide", page_title="AI-Powered Medical Report Assistant")

# --- Configuration & Setup ---
# Load environment variables from .env file
load_dotenv()

# Configure Google Generative AI (Gemini Pro)
# Make sure you have GOOGLE_API_KEY="YOUR_API_KEY" in your .env file
try:
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        st.error("Google Gemini Pro API key not found. Please set GOOGLE_API_KEY in your .env file.")
        st.stop() # Stop the app if API key is missing
    genai.configure(api_key=gemini_api_key)
except Exception as e:
    st.error(f"Error configuring Google Generative AI. Check your GOOGLE_API_KEY in .env: {e}")
    st.stop()

# Set Tesseract path if it's not in your system's PATH
# If you installed Tesseract to the default location, it might be auto-detected.
# If you get a 'TesseractNotFoundError', uncomment the line below and adjust the path.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Example for Windows

# Initialize EasyOCR reader (this can be resource-intensive, so do it once)
@st.cache_resource # This helps speed up the app by caching the loaded model
def load_easyocr_reader():
    """Loads the EasyOCR reader model once."""
    return easyocr.Reader(['en']) # Specify English language for OCR
reader = load_easyocr_reader()

# --- Helper Functions ---

def preprocess_image(image_bytes):
    """
    Preprocesses the input image (denoising, binarization) using OpenCV.
    """
    # Convert bytes to a numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_np is None:
        st.error("Could not decode image. Please check the file format.")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    # Denoising (optional, depends on image quality)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # Binarization (Otsu's thresholding)
    _, binarized = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binarized

def extract_text_with_ocr(image_bytes, ocr_tool="Tesseract"):
    """
    Extracts text from the preprocessed image using specified OCR tool (Tesseract or EasyOCR).
    Returns raw text content.
    """
    preprocessed_img = preprocess_image(image_bytes)
    if preprocessed_img is None:
        return ""

    if ocr_tool == "Tesseract":
        # Using image directly with pytesseract
        text = pytesseract.image_to_string(preprocessed_img)
        return text
    elif ocr_tool == "EasyOCR":
        # EasyOCR returns a list of bounding boxes, text, and confidence
        results = reader.readtext(preprocessed_img)
        text = "\n".join([res[1] for res in results]) # Concatenate all extracted text
        return text
    else:
        st.error("Invalid OCR tool specified. Choose 'Tesseract' or 'EasyOCR'.")
        return ""

def nlp_structure_data(raw_text):
    """
    Uses rule-based logic to structure the extracted content into a pandas DataFrame.
    Identifies test names, measured values, normal ranges, and units.
    Flags values outside the normal reference range.
    """
    structured_data = []
    lines = raw_text.split('\n')

    # Regex patterns for common medical lab report formats
    # This is a simplified example and might need extensive tuning for real-world reports
    # It tries to capture: Test Name (group 1) | Measured Value (group 2) | Unit (group 3) | Normal Range (group 4)
    # This pattern looks for "Test Name : Value Unit Range" or similar structures
    pattern = re.compile(
        r"^\s*([a-zA-Z\s\d,\(\)\-]+?)\s*[:\s]*([\d\.<>-]+)\s*([a-zA-Z%/\(\)\d\.]+)?\s*(?:(?:Normal|Range|Ref\.Range|Ref\.Value|Reference)\s*[:\s]*|\[|\(|\{)?\s*([\d\.<>-]+\s*[a-zA-Z%/\(\)\d\.]*)?\s*(?:\]|\)|\})?$"
    )

    for line in lines:
        match = pattern.match(line.strip())
        if match:
            test_name = match.group(1).strip()
            measured_value_str = match.group(2).strip()
            unit = match.group(3).strip() if match.group(3) else ""
            normal_range_str = match.group(4).strip() if match.group(4) else ""

            measured_value = None
            is_abnormal = False
            try:
                # Try to convert measured value to a float for comparison
                # Handles "<", ">" signs by removing them for conversion
                cleaned_measured_value = re.sub(r'[<> ]', '', measured_value_str)
                measured_value = float(cleaned_measured_value)
            except ValueError:
                measured_value = measured_value_str # Keep as string if cannot parse

            # Logic to identify values outside the normal range
            if isinstance(measured_value, (float, int)) and normal_range_str:
                try:
                    # Robust parsing for normal range string
                    if '-' in normal_range_str:
                        # Handles ranges like "10-20", "1.2-3.4"
                        parts = normal_range_str.split('-')
                        if len(parts) == 2:
                            lower_bound = float(parts[0].strip())
                            upper_bound = float(parts[1].strip())
                            if not (lower_bound <= measured_value <= upper_bound):
                                is_abnormal = True
                        else: # Handle cases like "10-20-30" if they exist, treat as string
                            pass
                    elif '<' in normal_range_str:
                        # Handles ranges like "<5", "< 100"
                        upper_bound = float(normal_range_str.replace('<', '').strip())
                        if measured_value >= upper_bound:
                            is_abnormal = True
                    elif '>' in normal_range_str:
                        # Handles ranges like ">10", "> 50"
                        lower_bound = float(normal_range_str.replace('>', '').strip())
                        if measured_value <= lower_bound:
                            is_abnormal = True
                    # Add more complex range parsing if needed (e.g., specific text like 'Negative', 'Present')
                except ValueError:
                    pass # If range parsing fails, assume normal for now or flag for manual review

            structured_data.append({
                "Test Name": test_name,
                "Measured Value": measured_value_str, # Store as string to preserve original format
                "Normal Range": normal_range_str,
                "Unit": unit,
                "Is Abnormal": is_abnormal
            })
    return pd.DataFrame(structured_data)

def generate_explanation(test_name, measured_value, normal_range, unit):
    """
    Uses Generative AI (Gemini Pro) to explain test results in simple language.
    """
    model = genai.GenerativeModel('gemini-pro')
    prompt = (
        f"Explain in simple language what it means if the patient's {test_name} is {measured_value} {unit}, "
        f"given the normal range is {normal_range}. Focus on a single test result. "
        f"Keep the explanation concise and easy to understand for a non-medical person."
    )
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating explanation: {e}. Please check your API key and internet connection."

def generate_risk_summary_and_suggestions(structured_data):
    """
    Generates a summary paragraph and a list of suggested actions based on extracted values.
    """
    abnormal_results = structured_data[structured_data["Is Abnormal"]]
    if abnormal_results.empty:
        return "All measured values appear to be within normal limits.", []

    summary_text_for_ai = "Abnormal test results:\n"
    for index, row in abnormal_results.iterrows():
        summary_text_for_ai += (
            f"- {row['Test Name']} is {row['Measured Value']} {row['Unit']} "
            f"(Normal Range: {row['Normal Range']}).\n"
        )

    # Use Generative AI for a more nuanced summary and suggestions
    model = genai.GenerativeModel('gemini-pro')
    prompt = (
        f"Analyze the following abnormal medical test results and provide a simple, short summary "
        f"and suggest general follow-up actions. Do not give specific medical advice or diagnoses. "
        f"Suggestions should be general, like 'Consult a doctor for further evaluation', 'Consider dietary changes', 'Monitor blood pressure'. "
        f"Format suggestions as a bulleted list. Start with a short summary paragraph, then the suggestions list.\n\n"
        f"Results:\n{summary_text_for_ai}\n\n"
        f"Summary and Suggested Actions:"
    )
    try:
        response = model.generate_content(prompt)
        full_response = response.text

        # Attempt to split summary from suggestions based on common patterns
        summary_paragraph = ""
        suggestions = []

        # Look for a common pattern for bulleted lists
        suggestion_list_start = re.search(r'^- ', full_response, re.MULTILINE)
        if suggestion_list_start:
            summary_paragraph = full_response[:suggestion_list_start.start()].strip()
            suggestions_raw = full_response[suggestion_list_start.start():].strip()
            suggestions = [line.strip() for line in suggestions_raw.split('\n- ') if line.strip()]
        else:
            # If no clear list, treat the whole response as summary and add a generic suggestion
            summary_paragraph = full_response.strip()
            suggestions = ["Consult your doctor for further evaluation."]

        return summary_paragraph, suggestions
    except Exception as e:
        return f"Error generating risk summary: {e}", ["Consult your doctor for further evaluation."]

# --- Streamlit UI ---

st.title("👨‍⚕️ AI-Powered Medical Report Assistant")
st.markdown("Upload your medical lab report (image or PDF) to get simple explanations of your test results.")

uploaded_file = st.file_uploader("Upload your medical report (JPEG, PNG, or PDF)", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    file_type = uploaded_file.type

    # Handle PDF files by converting to images first
    if file_type == "application/pdf":
        with st.spinner("Converting PDF to images..."):
            try:
                # Convert PDF bytes to PIL Images
                # poppler_path should be set automatically if poppler bin is in PATH
                images_from_pdf = convert_from_bytes(uploaded_file.read())
                st.success(f"PDF converted to {len(images_from_pdf)} image page(s).")
                # We'll process only the first page for simplicity here.
                # For multiple pages, you'd loop through images_from_pdf.
                if images_from_pdf:
                    # Convert PIL Image to bytes for consistent processing
                    img_byte_arr = io.BytesIO()
                    images_from_pdf[0].save(img_byte_arr, format='PNG')
                    image_bytes_to_process = img_byte_arr.getvalue()
                    st.image(image_bytes_to_process, caption="First Page of PDF", use_column_width=True)
                else:
                    st.error("Could not convert PDF to images.")
                    image_bytes_to_process = None
            except Exception as e:
                st.error(f"Error converting PDF: {e}. Make sure Poppler is installed and in your system's PATH.")
                image_bytes_to_process = None
    else: # It's an image file (JPEG, PNG)
        image_bytes_to_process = uploaded_file.read()
        st.image(image_bytes_to_process, caption="Uploaded Image", use_column_width=True)
        st.success("Image uploaded successfully!")

    if image_bytes_to_process:
        st.info("Extracting text using OCR. This might take a moment...")
        with st.spinner("Processing image and extracting text..."):
            # You can switch between "Tesseract" and "EasyOCR" here if one works better for your reports
            raw_text = extract_text_with_ocr(image_bytes_to_process, ocr_tool="Tesseract")
            st.subheader("Raw Extracted Text:")
            st.text(raw_text)

        st.info("Structuring data and identifying abnormal values...")
        with st.spinner("Analyzing text with NLP..."):
            structured_df = nlp_structure_data(raw_text)
            if not structured_df.empty:
                st.subheader("Structured Medical Data:")
                st.dataframe(structured_df)

                st.subheader("Explanation of Test Results:")
                abnormal_count = 0
                for index, row in structured_df.iterrows():
                    test_name = row["Test Name"]
                    measured_value = row["Measured Value"]
                    normal_range = row["Normal Range"]
                    unit = row["Unit"]
                    is_abnormal = row["Is Abnormal"]

                    if is_abnormal:
                        abnormal_count += 1
                        with st.expander(f"🔴 **{test_name}**: {measured_value} {unit} (Abnormal)"):
                            explanation = generate_explanation(test_name, measured_value, normal_range, unit)
                            st.write(explanation)
                    else:
                        with st.expander(f"🟢 **{test_name}**: {measured_value} {unit} (Normal)"):
                            explanation = generate_explanation(test_name, measured_value, normal_range, unit)
                            st.write(explanation)

                if abnormal_count > 0:
                    st.subheader("Risk Summary and Follow-up Suggestions:")
                    with st.spinner("Generating summary and suggestions..."):
                        summary, suggestions = generate_risk_summary_and_suggestions(structured_df)
                        st.markdown(summary)
                        if suggestions:
                            st.subheader("Suggested Actions:")
                            for suggestion in suggestions:
                                st.markdown(f"- {suggestion}")
                else:
                    st.subheader("All results appear to be within normal limits. Great!")

            else:
                st.warning("Could not extract structured data. The report format might be too complex or unclear. Please try a different report or improve image quality.")

    else:
        st.info("Please upload a medical report file to get started.")

st.sidebar.header("About")
st.sidebar.info(
    "This AI-powered assistant helps interpret medical lab reports by extracting data, "
    "structuring it, and explaining results in simple language using Generative AI. "
    "It can also suggest follow-up actions for abnormal values."
)
st.sidebar.markdown(
    "**Disclaimer:** This tool is for informational purposes only and does not constitute medical advice. "
    "Always consult a qualified healthcare professional for diagnosis and treatment."
)