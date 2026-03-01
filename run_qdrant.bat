@echo off
echo Starting Qdrant via Docker...
if not exist "%cd%\qdrant_storage" mkdir "%cd%\qdrant_storage"
docker run -d --name universal_rag_qdrant -p 6333:6333 -p 6334:6334 -v "%cd%\qdrant_storage:/qdrant/storage" --restart always qdrant/qdrant:latest
echo Qdrant is running.
pause
