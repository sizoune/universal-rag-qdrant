#!/bin/bash
echo "Stopping Qdrant container..."
sudo docker stop universal_rag_qdrant && sudo docker rm universal_rag_qdrant
echo "Qdrant container stopped and removed."
