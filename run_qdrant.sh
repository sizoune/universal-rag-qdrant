#!/bin/bash
echo "Starting Qdrant via Docker..."
mkdir -p $(pwd)/qdrant_storage
sudo docker run -d \
  --name universal_rag_qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  --restart always \
  qdrant/qdrant:latest
echo "Qdrant is running."
