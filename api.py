from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
import face_recognition
import numpy as np
import io
import os

app = FastAPI(title="Face Search Engine API")

# CONFIGURATION
COLLECTION_NAME = "faces"
IMAGE_FOLDER = "stored_images"

# Initialize Qdrant (Must point to the same path as ingestion)
qdrant = QdrantClient(path="qdrant_db")

# Mount the image folder so the frontend can access images via URL
# e.g., http://localhost:8000/static/my_photo.jpg
os.makedirs(IMAGE_FOLDER, exist_ok=True)
app.mount("/static", StaticFiles(directory=IMAGE_FOLDER), name="static")

@app.post("/search")
async def search_faces(file: UploadFile = File(...)):
    # 1. Read and Encode Input Image
    image_bytes = await file.read()
    image = face_recognition.load_image_file(io.BytesIO(image_bytes))
    input_encodings = face_recognition.face_encodings(image)

    if len(input_encodings) == 0:
        return {"message": "No faces detected in input image.", "matches": []}

    # Use the first face found
    target_encoding = input_encodings[0]

    # 2. Search in Qdrant
    # Qdrant handles the vector math efficiently
    search_result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query =target_encoding.tolist(),
        with_payload=True,
        limit=5,          # Return top 5 matches
        score_threshold=0.6 # Equivalent to (1 - 0.4 distance). Adjust based on results.
    ).points
    print(search_result)
    # 3. Format Results
    matches = []
    seen_files = set()

    for hit in search_result:
        if not hit.payload:
            continue
        filename = hit.payload.get("filename")
        
        # Deduplicate (if multiple faces from same image match)
        if filename not in seen_files:
            # Construct the URL pointing to this API's static mount
            # In production, replace 'localhost:8000' with your domain
            image_url = f"http://localhost:8000/static/{filename}"
            matches.append(image_url)
            seen_files.add(filename)

    return {
        "total_matches": len(matches),
        "matches": matches
    }