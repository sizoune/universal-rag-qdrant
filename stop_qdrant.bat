@echo off
echo Stopping Qdrant container...
docker stop universal_rag_qdrant
docker rm universal_rag_qdrant
echo Qdrant container stopped and removed.
pause
