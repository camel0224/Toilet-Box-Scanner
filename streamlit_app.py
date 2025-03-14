import streamlit as st
import cv2
import pytesseract
import pandas as pd
from datetime import datetime
import os
from PIL import Image
import numpy as np
import re
import io
import requests
from bs4 import BeautifulSoup
import asyncio
import aiohttp
from urllib.parse import quote_plus

# Set page config
st.set_page_config(page_title="Toilet Box Scanner", layout="wide")

# Initialize session state
if 'inventory_data' not in st.session_state:
    st.session_state.inventory_data = pd.DataFrame(
        columns=['Date', 'Model Number', 'Brand', 'Brand Model', 'Quantity', 'Notes', 'Image', 
                'Home Depot Price', 'Home Depot Link',
                'Lowes Price', 'Lowes Link',
                'Ferguson Price', 'Ferguson Link']
    )

# ... keep existing preprocess_image, extract_model_number functions ...

async def fetch_price(session, url, headers):
    try:
        async with session.get(url, headers=headers, timeout=10) as response:
            return await response.text()
    except:
        return None

async def search_product_prices(brand, model_number):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Prepare search terms
    search_term = f"{brand} {model_number}"
    encoded_search = quote_plus(search_term)
    
    # Define retailer URLs
    urls = {
        'homedepot': f'https://www.homedepot.com/s/{encoded_search}',
        'lowes': f'https://www.lowes.com/search?searchTerm={encoded_search}',
        'ferguson': f'https://www.ferguson.com/search/{encoded_search}'
    }
    
    results = {
        'Home Depot Price': 'N/A',
        'Home Depot Link': '',
        'Lowes Price': 'N/A',
        'Lowes Link': '',
        'Ferguson Price': 'N/A',
        'Ferguson Link': ''
    }
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for retailer, url in urls.items():
            task = fetch_price(session, url, headers)
            tasks.append((retailer, task))
        
        responses = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        for (retailer, _), html in zip(tasks, responses):
            if html and isinstance(html, str):
                soup = BeautifulSoup(html, 'html.parser')
                
                if retailer == 'homedepot':
                    results['Home Depot Link'] = urls['homedepot']
                    price_elem = soup.find('span', {'class': 'price'})
                    if price_elem:
                        results['Home Depot Price'] = price_elem.text.strip()
                
                elif retailer == 'lowes':
                    results['Lowes Link'] = urls['lowes']
                    price_elem = soup.find('span', {'class': 'price'})
                    if price_elem:
                        results['Lowes Price'] = price_elem.text.strip()
                
                elif retailer == 'ferguson':
                    results['Ferguson Link'] = urls['ferguson']
                    price_elem = soup.find('span', {'class': 'price'})
                    if price_elem:
                        results['Ferguson Price'] = price_elem.text.strip()
    
    return results

def process_image(image):
    # Preprocess the image
    processed = preprocess_image(image)
    
    # Try different PSM modes
    psm_modes = [11, 6, 3]  # 11: sparse text, 6: uniform block of text, 3: fully automatic
    best_result = ''
    
    for psm in psm_modes:
        # Configure Tesseract parameters
        custom_config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-'
        
        # Perform OCR
        text = pytesseract.image_to_string(processed, config=custom_config)
        
        # Extract model number
        model_number = extract_model_number(text)
        
        # If we found a valid model number, return it
        if re.match(r'\d{4}-\d{1,4}', model_number):
            return model_number
        
        # Keep the longest result as backup
        if len(model_number) > len(best_result):
            best_result = model_number
    
    return best_result

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
            brand = st.selectbox("Brand", ["Kohler", "TOTO", "Other"])
            brand_model = st.text_input("Brand Model", value=detected_text)
            quantity = st.number_input("Quantity Available", min_value=0, value=0)
            notes = st.text_area("Notes")
            
            submit_button = st.form_submit_button("Save to Inventory")
            
            if submit_button:
                with st.spinner('Fetching prices from retailers...'):
                    # Get prices from retailers
                    prices = asyncio.run(search_product_prices(brand, brand_model))
                
                # Add to inventory
                new_row = {
                    'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Model Number': detected_text,
                    'Brand': brand,
                    'Brand Model': brand_model,
                    'Quantity': quantity,
                    'Notes': notes,
                    'Image': uploaded_file.getvalue(),
                    **prices  # Add all the price and link information
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
                col1, col2, col3 = st.columns([1, 1, 1])
                
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
                
                with col3:
                    st.subheader("Retail Prices")
                    if row['Home Depot Link']:
                        st.write(f"**Home Depot:** {row['Home Depot Price']}")
                        st.write(f"[View at Home Depot]({row['Home Depot Link']})")
                    
                    if row['Lowes Link']:
                        st.write(f"**Lowes:** {row['Lowes Price']}")
                        st.write(f"[View at Lowes]({row['Lowes Link']})")
                    
                    if row['Ferguson Link']:
                        st.write(f"**Ferguson:** {row['Ferguson Price']}")
                        st.write(f"[View at Ferguson]({row['Ferguson Link']})")
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
