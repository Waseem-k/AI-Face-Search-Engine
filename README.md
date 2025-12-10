# AI Face Search Engine 🔍

A full-stack AI application that performs facial recognition and similarity search on large datasets. It uses **Vector Embeddings** to find matches in milliseconds, scalable to millions of images.

## 🚀 Project Overview

This system allows users to upload a photo of a person and instantly find other photos of that same person from a database. Unlike simple filename matching, this uses **Deep Learning** to generate 128-dimensional vector embeddings for every face, allowing it to recognize people even with different lighting, poses, or expressions.

### Key Features
* **Vector Search:** Uses **Qdrant** (Vector Database) for high-performance similarity search.
* **Face Embeddings:** powered by `face_recognition` (dlib) to generate state-of-the-art face encodings.
* **Scalable Architecture:** Decoupled Microservices architecture (Ingestion, API, Frontend).
* **Smart Ingestion:** ETL pipeline that processes raw images, handles deduplication, and indexes metadata.
* **Interactive UI:** A clean frontend built with **Streamlit** for real-time interaction.

---

## 🛠️ Tech Stack

| Component | Technology | Why this choice? |
| :--- | :--- | :--- |
| **Database** | **Qdrant** | Production-grade Vector DB (Rust-based), faster and more scalable than storing numpy arrays. |
| **Backend** | **FastAPI** | High-performance async API to serve model inference and static files. |
| **Frontend** | **Streamlit** | Rapid prototyping UI for data apps; handles file uploads and state management seamlessly. |
| **ML Model** | **dlib (HOG/CNN)** | Industry standard for face detection and encoding (99.38% accuracy on LFW benchmark). |
| **Dataset** | **CelebA** | Large-scale dataset (200k+ images) used to demonstrate "Wild" face recognition capabilities. |

---

## 📂 Project Structure

```text
├── stored_images/       # The "S3 Bucket" (Local storage for indexed images)
├── qdrant_db/           # Local vector database files (Auto-generated)
├── raw_data/            # Staging area for raw datasets (e.g., CelebA)
├── ingestion.py         # ETL pipeline: Detects faces -> Embeds -> Upserts to Qdrant
├── api.py               # FastAPI Backend: Handles search requests & file serving
├── frontend_streamlit.py# Streamlit Frontend: User Interface
├── prepare_data.py      # Utility: Samples and cleans the raw CelebA dataset
└── requirements.txt     # Python dependencies
```

⚡ Getting Started
1. Prerequisites
Python 3.9+

Visual Studio C++ Build Tools (Windows only, required for dlib)

2. Installation
Clone the repository and install dependencies:

Bash

git clone [https://github.com/yourusername/face-search-engine.git](https://github.com/yourusername/face-search-engine.git)
cd face-search-engine

# Create virtual environment
python -m venv venv
# Activate: source venv/bin/activate (Mac/Linux) or venv\Scripts\activate (Windows)

pip install -r requirements.txt
3. Data Setup (The ETL Pipeline)
This project uses the CelebA dataset.

Download img_align_celeba.zip and identity_CelebA.txt from Kaggle.

Extract them into a raw_data/ folder.

Run the preparation script to sample and clean the data:

Bash

python prepare_data.py
Run the ingestion script to generate embeddings and populate the Vector DB:

Bash

python ingestion.py
4. Running the Application
You need two terminal windows running simultaneously.

Terminal 1: The API Backend

Bash

uvicorn api:app --reload --port 8000
Runs on http://localhost:8000

Terminal 2: The Frontend

Bash

streamlit run frontend_streamlit.py
Opens in browser at http://localhost:8501

📸 Demo
Open the Streamlit UI.

Upload a photo of a celebrity (e.g., from Google Images).

The system converts the face to a vector, queries Qdrant, and returns the top 5 closest matches from the dataset.

🧠 How It Works (Under the Hood)
Preprocessing: When an image is ingested, the system uses HOG (Histogram of Oriented Gradients) to locate the face.

Encoding: The face pixels are mapped to a 128-dimensional vector space where similar faces are close together.

Indexing: These vectors are stored in Qdrant with a payload containing the filename and metadata.

Search: When you upload a query image, it is converted to a vector. We calculate the Euclidean Distance between the query vector and all vectors in the DB.

Ranking: Results with a distance score < 0.6 (threshold) are returned as matches.

🔮 Future Improvements
Dockerization: Containerize the API and Frontend for easy deployment.

Cloud Storage: Replace local stored_images with AWS S3.

GPU Acceleration: Enable CNN-based face detection for higher accuracy.

Author

Waseem Ashraf Khan, AI/ML Engineer, https://www.linkedin.com/in/waseem-khan-data-scientist/ 
