from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    question: str = Field(..., description="User question")
    session_id: str | None = Field(default="default", description="Chat session id")
    system_prompt: str | None = Field(
        default=None,
        max_length=8000,
        description="Extra system instructions appended to the base prompt for this request",
    )
    enable_web_search: bool = Field(
        default=False,
        description="Aktifkan fallback web search bila jawaban tak ada di dokumen",
    )


class RetrieveRequest(BaseModel):
    question: str = Field(..., description="User question to retrieve against")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Override MAX_SEARCH_RESULTS for this request",
    )


class IngestWebRequest(BaseModel):
    url: str


class IngestUrlRequest(BaseModel):
    """Ingest a remote file (e.g. presigned S3 URL) into the caller's namespace."""

    url: str
    source: str | None = Field(
        default=None,
        description="Stable source id stamped on chunks; defaults to the URL",
    )
    source_type: str | None = Field(
        default="remote",
        description="source_type metadata (e.g. remote, ppid_document)",
    )


class IngestPathRequest(BaseModel):
    path: str


class IngestStatusResponse(BaseModel):
    running: bool
    current_task: str | None = None
    current_source: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    last_message: str | None = None


class TokenUsage(BaseModel):
    input_estimate: int
    output_estimate: int
    total_estimate: int


class LocationItem(BaseModel):
    """Single citation location within a source document.

    `display` is a pre-formatted Indonesian label (e.g. "Halaman 5 — Bab 2").
    Structured fields (`page`, `url_fragment`, `line_range`) are optional and
    used by clients for deep-linking only.
    """

    model_config = ConfigDict(frozen=True)

    display: str
    chunk_preview: str
    page: int | None = None
    url_fragment: str | None = None
    line_range: tuple[int, int] | None = None


class SourceItem(BaseModel):
    """A source document referenced in an answer, with one or more cited locations."""

    model_config = ConfigDict(frozen=True)

    source: str
    source_type: str
    filename: str | None = None
    locations: list[LocationItem]


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    token_usage: TokenUsage
    elapsed_ms: int | None = None
    web_search_used: bool = False


class RetrievedChunk(BaseModel):
    text: str
    source: str
    source_type: str = "local"
    filename: str | None = None
    page: int | None = None
    heading_path: list[str] | None = None
    chunk_kind: str | None = None
    namespace: str | None = None
    score: float | None = None
    display: str | None = None


class RetrieveResponse(BaseModel):
    chunks: list[RetrievedChunk]
    sources: list[SourceItem]
    elapsed_ms: int | None = None


class FileItem(BaseModel):
    source_id: str
    source: str
    source_type: str
    chunk_count: int
    last_seen: str | None = None
    in_s3: bool = False


class FileListResponse(BaseModel):
    items: list[FileItem]
    total: int
    page: int = 1
    page_size: int = 10
    total_pages: int = 1


class UploadFileItem(BaseModel):
    upload_id: str
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
