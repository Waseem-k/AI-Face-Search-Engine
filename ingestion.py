import os
import face_recognition
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid
import logging

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# CONFIGURATION
IMAGE_FOLDER = "stored_images"  # Create this folder and put your photos here
COLLECTION_NAME = "faces"

def run_ingestion():
    # 1. Initialize Qdrant (Local Mode - saves to disk)
    # This creates a local 'qdrant_db' folder to store data
    client = QdrantClient(path="qdrant_db") 
    
    # 2. Create Collection if it doesn't exist
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=128, distance=Distance.COSINE)  # Face embeddings are 128-dim,
        )
        logger.info(f"Created collection '{COLLECTION_NAME}'")

    # 3. Process Images
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        logger.info(f"Created folder '{IMAGE_FOLDER}'. Please add images and run again.")
        return

    logger.info("Scanning images...")
    
    # Get existing files to avoid re-processing (simple check)
    # In a real app, you'd check Qdrant for existing IDs, but here we just process all for simplicity
    # or you can implement a simple 'processed.txt' logic like before.
    
    points_to_upsert = []
    
    for root, dirs, files in os.walk(IMAGE_FOLDER):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, filename)
                
                try:
                    # Load Image
                    image = face_recognition.load_image_file(file_path)
                    
                    # Detect Faces (HOG is faster, CNN is more accurate)
                    face_locations = face_recognition.face_locations(image, model="hog")
                    face_encodings = face_recognition.face_encodings(image, face_locations)
                    
                    if face_encodings:
                        logger.info(f"Found {len(face_encodings)} faces in {filename}")
                        
                        for i, encoding in enumerate(face_encodings):
                            # Create a unique ID for the point
                            point_id = str(uuid.uuid4())
                            
                            person_label = "_".join(filename.split("_")[:2]) 
                            
                            payload = {
                                "filename": filename,
                                "person_label": person_label, # Storing the ID as a tag
                                "face_index": i,
                                "path": file_path
                            }
                            # Add to batch
                            points_to_upsert.append(PointStruct(
                                id=point_id,
                                vector=encoding.tolist(),
                                payload=payload
                            ))
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")

    # 4. Upload to Qdrant
    if points_to_upsert:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_to_upsert
        )
        logger.info(f"Successfully indexed {len(points_to_upsert)} faces to Qdrant.")
    else:
        logger.info("No new faces found to index.")

if __name__ == "__main__":
    run_ingestion()