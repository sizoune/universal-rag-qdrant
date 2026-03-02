from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., description="User question")
    session_id: str | None = Field(default="default", description="Chat session id")


class IngestWebRequest(BaseModel):
    url: str


class IngestPathRequest(BaseModel):
    path: str


class TokenUsage(BaseModel):
    input_estimate: int
    output_estimate: int
    total_estimate: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    token_usage: TokenUsage


class FileItem(BaseModel):
    source_id: str
    source: str
    source_type: str
    chunk_count: int
    last_seen: str | None = None


class FileListResponse(BaseModel):
    items: list[FileItem]
    total: int


class UploadFileItem(BaseModel):
    filename: str
    path: str
    size_bytes: int
    modified_at: str
    ingested: bool
    ingest_status: str
    source_id: str | None = None
    chunk_count: int | None = None
    last_seen: str | None = None


class UploadFileListResponse(BaseModel):
    items: list[UploadFileItem]
    total: int
    page: int
    page_size: int
    total_pages: int
    uploads_dir: str


class OperationResponse(BaseModel):
    success: bool
    message: str
    deleted_chunks: int | None = None
    added_chunks: int | None = None
    skipped: bool | None = None
    uploads_dir: str | None = None
    processed_files: int | None = None
