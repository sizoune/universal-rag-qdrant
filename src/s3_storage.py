"""S3 storage helper for uploaded files.

When S3_BUCKET is configured, uploaded files are moved to S3 after ingestion
and downloaded back from S3 when reingest is needed.
"""

import logging
import os
import tempfile

import boto3
from botocore.exceptions import ClientError

from src.config import config

logger = logging.getLogger(__name__)


def is_s3_enabled() -> bool:
    return bool(config.S3_BUCKET)


def _get_client():
    kwargs: dict = {}
    if config.S3_ENDPOINT_URL:
        kwargs["endpoint_url"] = config.S3_ENDPOINT_URL
    if config.S3_REGION:
        kwargs["region_name"] = config.S3_REGION
    return boto3.client("s3", **kwargs)


def _s3_key(filename: str) -> str:
    prefix = config.S3_PREFIX.strip("/")
    if prefix:
        return f"{prefix}/{filename}"
    return filename


def upload_to_s3(local_path: str) -> str:
    """Upload a local file to S3. Returns the S3 key."""
    client = _get_client()
    filename = os.path.basename(local_path)
    key = _s3_key(filename)
    client.upload_file(local_path, config.S3_BUCKET, key)
    logger.info("Uploaded %s to s3://%s/%s", local_path, config.S3_BUCKET, key)
    return key


def download_from_s3(s3_key: str, dest_dir: str | None = None) -> str:
    """Download a file from S3 to a temp directory. Returns the local path."""
    client = _get_client()
    filename = os.path.basename(s3_key)
    if dest_dir is None:
        dest_dir = tempfile.mkdtemp(prefix="rag_s3_")
    os.makedirs(dest_dir, exist_ok=True)
    local_path = os.path.join(dest_dir, filename)
    client.download_file(config.S3_BUCKET, s3_key, local_path)
    logger.info("Downloaded s3://%s/%s to %s", config.S3_BUCKET, s3_key, local_path)
    return local_path


def delete_from_s3(s3_key: str) -> bool:
    """Delete a file from S3. Returns True if successful."""
    client = _get_client()
    try:
        client.delete_object(Bucket=config.S3_BUCKET, Key=s3_key)
        logger.info("Deleted s3://%s/%s", config.S3_BUCKET, s3_key)
        return True
    except ClientError:
        logger.exception("Failed to delete s3://%s/%s", config.S3_BUCKET, s3_key)
        return False


def file_exists_in_s3(s3_key: str) -> bool:
    """Check if a file exists in S3."""
    client = _get_client()
    try:
        client.head_object(Bucket=config.S3_BUCKET, Key=s3_key)
        return True
    except ClientError:
        return False


def get_s3_key_for_source(source_path: str) -> str:
    """Derive the S3 key from a local source path (uses the basename)."""
    return _s3_key(os.path.basename(source_path))


def list_s3_files() -> list[dict]:
    """List all files under the configured prefix. Returns list of {key, size, last_modified}."""
    client = _get_client()
    prefix = config.S3_PREFIX.strip("/")
    if prefix:
        prefix += "/"

    result = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=config.S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            result.append({
                "key": obj["Key"],
                "filename": os.path.basename(obj["Key"]),
                "size_bytes": obj["Size"],
                "last_modified": obj["LastModified"].isoformat(),
            })
    return result
