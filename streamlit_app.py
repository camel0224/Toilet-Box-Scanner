import streamlit as st
import cv2
import pytesseract
import pandas as pd
from datetime import datetime
import os
from PIL import Image
import numpy as np
import io

# Set page config
st.set_page_config(page_title="Toilet Box Scanner", layout="wide")

# Initialize session state
if 'inventory_data' not in st.session_state:
    st.session_state.inventory_data = pd.DataFrame(
        columns=['Date', 'Model Number', 'Brand', 'Brand Model', 'Quantity', 'Notes', 'Image']
    )

def process_image(image):
    # Convert to OpenCV format for OCR
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Perform OCR
    text = pytesseract.image_to_string(cv_image)
    return text.strip()

# Title
st.title("Toilet Box Scanner")

# Tabs
tab1, tab2 = st.tabs(["Add Item", "View Inventory"])

with tab1:
    st.header("Add New Item")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a photo of the toilet box", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process image with OCR
        detected_text = process_image(image)
        st.write("Detected Model Number:", detected_text)
        
        # Form for additional details
        with st.form("inventory_form"):
            brand = st.text_input("Brand")
            brand_model = st.text_input("Brand Model")
            quantity = st.number_input("Quantity Available", min_value=0, value=0)
            notes = st.text_area("Notes")
            
            submit_button = st.form_submit_button("Save to Inventory")
            
            if submit_button:
                # Add to inventory
                new_row = {
                    'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Model Number': detected_text,
                    'Brand': brand,
                    'Brand Model': brand_model,
                    'Quantity': quantity,
                    'Notes': notes,
                    'Image': uploaded_file.getvalue()
                }
                
                st.session_state.inventory_data = pd.concat([
                    st.session_state.inventory_data,
                    pd.DataFrame([new_row])
                ], ignore_index=True)
                
                st.success("Item added to inventory!")
                
                # Clear the form (by rerunning the app)
                st.experimental_rerun()

with tab2:
    st.header("Inventory List")
    
    if len(st.session_state.inventory_data) > 0:
        # Display inventory table
        for index, row in st.session_state.inventory_data.iterrows():
            with st.expander(f"Item {index + 1} - {row['Brand']} {row['Brand Model']}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if row['Image'] is not None:
                        image = Image.open(io.BytesIO(row['Image']))
                        st.image(image, caption="Product Image", use_column_width=True)
                
                with col2:
                    st.write(f"**Date:** {row['Date']}")
                    st.write(f"**Model Number:** {row['Model Number']}")
                    st.write(f"**Brand:** {row['Brand']}")
                    st.write(f"**Brand Model:** {row['Brand Model']}")
                    st.write(f"**Quantity:** {row['Quantity']}")
                    st.write(f"**Notes:** {row['Notes']}")
    else:
        st.info("No items in inventory yet.")

    # Download button for inventory
    if len(st.session_state.inventory_data) > 0:
        csv = st.session_state.inventory_data.drop('Image', axis=1).to_csv(index=False)
        st.download_button(
            label="Download Inventory as CSV",
            data=csv,
            file_name="inventory.csv",
            mime="text/csv"
        ) 