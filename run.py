import os
import json
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory="C:/Users/hi/dev/real_pro/recognize-anything/images"), name="images")

# Directory for image files
images_dir = r"C:\Users\hi\dev\real_pro\recognize-anything\images\screenshot"

# Load image tags data from JSON file
with open("image_tags.json", "r") as f:
    image_tags = json.load(f)

# Combine tags for each image into a single document
tag_documents = [" ".join(tags) for tags in image_tags.values()]
print(f"tag_documents0!!!:{tag_documents}")

# Ensure tag documents are not empty
if not tag_documents or all(not doc for doc in tag_documents):
    raise ValueError("태그 문서가 비어있습니다. 태그를 확인하세요.")

# Create Sentence-BERT embeddings
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
sbert_embeddings = sbert_model.encode(tag_documents)

# Perform KMeans clustering
n_clusters = 4  # Set appropriate cluster count
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(sbert_embeddings)

# Define category names based on cluster numbers
category_map = {0: "지도", 1: "음식", 2: "패션", 3: "동물"}

# Organize data into a DataFrame with categories
df = pd.DataFrame({
    'image': list(image_tags.keys()), 
    'cluster': clusters, 
    'tags': tag_documents
})
df['category'] = df['cluster'].map(category_map)

# FastAPI upload endpoint
@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    # Save uploaded files
    uploaded_image_paths = []
    for file in files:
        file_location = os.path.join(images_dir, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())
        # Save uploaded image URLs
        uploaded_image_paths.append(f"http://localhost:8000/images/screenshot/{file.filename}")
    
    # Reload image tags data from JSON file
    with open("image_tags.json", "r") as f:
        image_tags = json.load(f)

    # Combine tags into a single document
    tag_documents = [" ".join(tags) for tags in image_tags.values()]
    print(f"tag_documents1:{tag_documents}")

    if not tag_documents or all(not doc for doc in tag_documents):
        return JSONResponse(content={"error": "태그 문서가 비어있습니다."}, status_code=400)

    # Generate embeddings using Sentence-BERT
    sbert_embeddings = sbert_model.encode(tag_documents)

    # Re-run KMeans clustering
    clusters = kmeans.fit_predict(sbert_embeddings)

    print(f"tag_documents2:{tag_documents}\n")
    # Update DataFrame with new clustering results
    df = pd.DataFrame({
        'image': list(image_tags.keys()), 
        'cluster': clusters, 
        'tags': tag_documents
    })
    df['category'] = df['cluster'].map(category_map)  # Map clusters to categories

    # Handle cases where uploaded image URLs are less than the DataFrame length
    if len(uploaded_image_paths) < len(df):
        uploaded_image_paths += [None] * (len(df) - len(uploaded_image_paths))

    # Add image URLs to the DataFrame
    df['image_url'] = uploaded_image_paths[:len(df)]

    # Check cluster count and distribution
    cluster_counts = df['category'].value_counts().to_dict()
    print(f"df:{df.head()}")

    # Return results
    return {
        "results": df.to_dict(orient="records"),
        "cluster_counts": cluster_counts
    }

# FastAPI run code
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
