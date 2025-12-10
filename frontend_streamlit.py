import streamlit as st
import requests
from PIL import Image

# CONFIGURATION
API_URL = "http://localhost:8000/search"

st.set_page_config(page_title="Face Search Engine", layout="wide")

st.title("AI Face Search Engine")
st.markdown("""
Upload a photo of a person to find other photos of them in the database.
""")

st.divider()

# Layout: Two columns (Upload vs Results)
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Search Button
        if st.button("Search Database", type="primary"):
            with st.spinner("Searching Vector Database..."):
                try:
                    # Reset pointer to beginning of file so it can be read again
                    uploaded_file.seek(0)
                    
                    # Send POST request to FastAPI
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.results = data.get("matches", [])
                        st.session_state.count = data.get("total_matches", 0)
                        st.success(f"Search Complete! Found {st.session_state.count} matches.")
                    else:
                        st.error(f"Error: API returned status {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Connection Error: {e}")

with col2:
    st.subheader("2. Search Results")
    
    if "results" in st.session_state and st.session_state.results:
        # Create a grid of images
        cols = st.columns(3) # 3 images per row
        for idx, img_url in enumerate(st.session_state.results):
            with cols[idx % 3]:
                st.image(img_url, use_container_width=True, caption=f"Match {idx+1}")
    else:
        st.info("No results to display yet. Upload an image to start.")