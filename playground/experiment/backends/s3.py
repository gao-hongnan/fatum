from __future__ import annotations

import json
import platform
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from fatum.experiment.types import StorageKey

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client


class SystemInfo(BaseModel):
    """System information for experiment tracking."""

    platform: str = Field(default_factory=lambda: platform.system())
    platform_release: str = Field(default_factory=lambda: platform.release())
    platform_version: str = Field(default_factory=lambda: platform.version())
    architecture: str = Field(default_factory=lambda: platform.machine())
    python_version: str = Field(default_factory=lambda: platform.python_version())
    node: str = Field(default_factory=lambda: platform.node())


class S3Config(BaseModel):
    model_config = ConfigDict(extra="ignore")

    bucket: str = Field(description="S3 bucket name")

    endpoint_url: str | None = Field(default=None, description="S3 endpoint URL")
    region_name: str = Field(default="us-east-1", description="AWS region")
    aws_access_key_id: str | None = Field(default=None, description="AWS access key")
    aws_secret_access_key: str | None = Field(default=None, description="AWS secret key")
    aws_session_token: str | None = Field(default=None, description="AWS session token")

    signature_version: str = Field(default="s3v4", description="Signature version")
    addressing_style: Literal["path", "virtual", "auto"] = Field(default="virtual", description="S3 addressing style")
    use_ssl: bool = Field(default=True, description="Use SSL/TLS")
    verify: bool | str = Field(default=True, description="SSL certificate verification")

    server_side_encryption: Literal["AES256", "aws:kms"] | None = Field(
        default="AES256", description="Server-side encryption method"
    )
    sse_kms_key_id: str | None = Field(default=None, description="KMS key ID for encryption")
    sse_customer_algorithm: str | None = Field(default=None, description="Customer encryption algorithm")
    sse_customer_key: str | None = Field(default=None, description="Customer encryption key")

    max_pool_connections: int = Field(default=10, ge=1, description="Maximum connection pool size")
    max_retry_attempts: int = Field(default=3, ge=0, description="Maximum retry attempts")
    retry_mode: Literal["legacy", "standard", "adaptive"] = Field(default="adaptive", description="Retry mode")
    read_timeout: float = Field(default=60.0, gt=0, description="Read timeout in seconds")
    connect_timeout: float = Field(default=60.0, gt=0, description="Connect timeout in seconds")

    multipart_threshold_mb: int = Field(default=8, ge=1, description="Multipart threshold in MB")
    multipart_chunksize_mb: int = Field(default=8, ge=1, description="Multipart chunk size in MB")
    max_concurrency: int = Field(default=10, ge=1, description="Max concurrent transfers")
    use_threads: bool = Field(default=True, description="Use threads for transfers")
    max_bandwidth: int | None = Field(default=None, ge=1, description="Max bandwidth in KB/s")

    cache_dir: Path = Field(default_factory=lambda: Path("./cache"), description="Local cache directory")
    auto_create_bucket: bool = Field(default=True, description="Auto-create bucket if missing")
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [".DS_Store", "Thumbs.db", "desktop.ini", "*.pyc", "__pycache__"],
        description="File patterns to exclude from uploads",
    )

    @field_validator("cache_dir", mode="before")
    @classmethod
    def validate_cache_dir(cls, v: Any) -> Path:
        """Convert string to Path if needed."""
        return Path(v) if isinstance(v, str) else v

    def get_boto_config(self) -> Config:
        """Build botocore Config object from settings."""
        return Config(
            max_pool_connections=self.max_pool_connections,
            retries={"max_attempts": self.max_retry_attempts, "mode": self.retry_mode},
            s3={"addressing_style": self.addressing_style, "signature_version": self.signature_version},  # type: ignore[arg-type]
            read_timeout=self.read_timeout,
            connect_timeout=self.connect_timeout,
        )

    def get_transfer_config(self) -> TransferConfig:
        """Build TransferConfig object from settings."""
        return TransferConfig(
            multipart_threshold=self.multipart_threshold_mb * 1024 * 1024,
            multipart_chunksize=self.multipart_chunksize_mb * 1024 * 1024,
            max_concurrency=self.max_concurrency,
            use_threads=self.use_threads,
            max_bandwidth=self.max_bandwidth * 1024 if self.max_bandwidth else None,
        )

    def get_client_kwargs(self) -> dict[str, Any]:
        """Build kwargs for boto3.client() from settings."""
        kwargs: dict[str, Any] = {
            "region_name": self.region_name,
            "use_ssl": self.use_ssl,
            "verify": self.verify,
        }

        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        if self.aws_access_key_id:
            kwargs["aws_access_key_id"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            kwargs["aws_secret_access_key"] = self.aws_secret_access_key
        if self.aws_session_token:
            kwargs["aws_session_token"] = self.aws_session_token

        return kwargs

    def get_extra_args(self) -> dict[str, Any]:
        """Build ExtraArgs for upload/download operations."""
        extra_args: dict[str, Any] = {}

        if self.server_side_encryption:
            extra_args["ServerSideEncryption"] = self.server_side_encryption
        if self.sse_kms_key_id:
            extra_args["SSEKMSKeyId"] = self.sse_kms_key_id
        if self.sse_customer_algorithm:
            extra_args["SSECustomerAlgorithm"] = self.sse_customer_algorithm
        if self.sse_customer_key:
            extra_args["SSECustomerKey"] = self.sse_customer_key

        return extra_args


class S3StorageError(Exception):
    """Base exception for S3 storage operations."""

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = context or {}


class S3ConnectionError(S3StorageError):
    """Raised when S3 connection fails."""

    pass


class S3BucketError(S3StorageError):
    """Raised for bucket-related errors."""

    pass


class S3ObjectNotFoundError(S3StorageError):
    """Raised when S3 object is not found."""

    pass


class S3Storage:
    def __init__(
        self,
        config: S3Config,
        client: S3Client | None = None,
    ) -> None:
        self.config = config
        self.cache_dir = config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = client
        self._owns_client = client is None
        if self._client is None:
            self._client = self._create_client()

        self._transfer_config = config.get_transfer_config()
        self._bucket_verified = False

        self._run_id: str | None = None
        self._experiment_id: str | None = None
        if config.auto_create_bucket:
            self._ensure_bucket()

    def _create_client(self) -> S3Client:
        try:
            boto_config = self.config.get_boto_config()
            client_kwargs = self.config.get_client_kwargs()
            return boto3.client("s3", config=boto_config, **client_kwargs)
        except NoCredentialsError as e:
            raise S3ConnectionError("No AWS credentials found", {"error": str(e)}) from e
        except Exception as e:
            raise S3ConnectionError("Failed to create S3 client", {"error": str(e)}) from e

    @property
    def client(self) -> S3Client:
        if self._client is None:
            raise S3ConnectionError("S3 client not initialized")
        return self._client

    def _ensure_bucket(self) -> None:
        if self._bucket_verified:
            return

        try:
            self.client.head_bucket(Bucket=self.config.bucket)
            self._bucket_verified = True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                if self._owns_client and self.config.auto_create_bucket:
                    try:
                        self.client.create_bucket(Bucket=self.config.bucket)
                        self._bucket_verified = True
                    except ClientError as create_error:
                        raise S3BucketError(f"Failed to create bucket '{self.config.bucket}'") from create_error
                else:
                    raise S3BucketError(f"Bucket '{self.config.bucket}' does not exist") from e
            else:
                raise S3ConnectionError(f"Failed to access bucket '{self.config.bucket}'") from e

    def initialize(self, run_id: str, experiment_id: str) -> None:
        self._run_id = run_id
        self._experiment_id = experiment_id

    def finalize(self, status: str) -> None:
        pass

    @property
    def run_id(self) -> str | None:
        """Get current run ID if set."""
        return self._run_id

    @property
    def experiment_id(self) -> str | None:
        """Get current experiment ID if set."""
        return self._experiment_id

    def _upload_file(self, key: str, path: Path) -> str:
        extra_args = self.config.get_extra_args()

        self.client.upload_file(
            str(path),
            self.config.bucket,
            key,
            Config=self._transfer_config,
            ExtraArgs=extra_args if extra_args else None,
        )
        return f"s3://{self.config.bucket}/{key}"

    def _upload_directory(self, prefix: str, directory: Path) -> dict[str, str]:
        files = [
            (f, f"{prefix}/{f.relative_to(directory)}".replace("\\", "/"))
            for f in directory.rglob("*")
            if f.is_file() and not any(f.match(p) for p in self.config.exclude_patterns)
        ]

        if not files:
            return {}

        results: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as executor:
            futures = {
                executor.submit(self._upload_file, s3_key, local_path): (str(local_path), s3_key)
                for local_path, s3_key in files
            }

            for future in as_completed(futures):
                local_path_str, _ = futures[future]
                try:
                    uri = future.result()
                    results[local_path_str] = uri
                except Exception as e:
                    print(f"Failed to upload {local_path_str}: {e}")

        return results

    def _download_file(self, key: str, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)

        response = self.client.head_object(Bucket=self.config.bucket, Key=key)
        size = response.get("ContentLength", 0)

        if size > self._transfer_config.multipart_threshold:
            self.client.download_file(self.config.bucket, key, str(target), Config=self._transfer_config)
        else:
            obj = self.client.get_object(Bucket=self.config.bucket, Key=key)
            target.write_bytes(obj["Body"].read())

        return target

    def _download_bytes(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.config.bucket, Key=key)
        return response["Body"].read()

    def _download_directory(self, prefix: str, target_dir: Path, objects: list[Any]) -> dict[str, Path]:
        target_dir.mkdir(parents=True, exist_ok=True)

        downloads = []
        for obj in objects:
            s3_key = obj["Key"]
            relative = s3_key[len(str(prefix)) :].lstrip("/")
            if relative:
                local_path = target_dir / relative
                downloads.append((s3_key, local_path))

        results = {}
        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as executor:
            futures = {
                executor.submit(self._download_file, s3_key, local_path): (s3_key, local_path)
                for s3_key, local_path in downloads
            }

            for future in as_completed(futures):
                s3_key, local_path = futures[future]
                try:
                    path = future.result()
                    results[s3_key] = path
                except Exception as e:
                    print(f"Failed to download {s3_key}: {e}")

        return results

    def _list_objects(self, prefix: str) -> list[Any]:
        paginator = self.client.get_paginator("list_objects_v2")
        objects = []

        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
            if "Contents" in page:
                objects.extend(page["Contents"])

        return objects

    def save(
        self,
        key: str | StorageKey,
        data: Path | str | bytes | dict[str, Any],
    ) -> str | dict[str, str]:
        self._ensure_bucket()
        key = str(key)

        if isinstance(data, Path):
            if not data.exists():
                raise FileNotFoundError(f"Source path not found: {data}")

            if data.is_dir():
                return self._upload_directory(key, data)
            else:
                return self._upload_file(key, data)

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp:
            temp_path = Path(tmp.name)
            try:
                if isinstance(data, str):
                    tmp.write(data.encode())
                elif isinstance(data, bytes):
                    tmp.write(data)
                elif isinstance(data, dict):
                    tmp.write(json.dumps(data, indent=2, default=str).encode())
                else:
                    raise TypeError(f"Unsupported data type: {type(data).__name__}. Use Path, str, bytes, or dict")

                tmp.flush()
                return self._upload_file(key, temp_path)
            finally:
                temp_path.unlink(missing_ok=True)

    def load(
        self,
        key: str | StorageKey,
        *,
        target: Path | None = None,
        as_type: Literal["bytes", "str", "json", "auto"] = "auto",
    ) -> Path | dict[str, Path] | bytes | str | dict[str, Any] | list[Any]:
        self._ensure_bucket()
        key = str(key)

        objects = self._list_objects(key)

        if len(objects) == 0:
            raise S3ObjectNotFoundError(f"Key not found: {key}")

        is_directory = len(objects) > 1 or (len(objects) == 1 and objects[0]["Key"] != key)

        if is_directory:
            target_dir = target or self.cache_dir / key
            return self._download_directory(key, target_dir, objects)

        if target:
            return self._download_file(key, target)

        data = self._download_bytes(key)

        if as_type == "bytes":
            return data
        elif as_type == "str":
            return data.decode()
        elif as_type == "json":
            result = json.loads(data)
            return result  # type: ignore[no-any-return]
        else:
            try:
                result = json.loads(data)
                return result  # type: ignore[no-any-return]
            except (json.JSONDecodeError, UnicodeDecodeError):
                try:
                    return data.decode()
                except UnicodeDecodeError:
                    return data

    def exists(self, key: StorageKey) -> bool:
        self._ensure_bucket()

        try:
            self.client.head_object(Bucket=self.config.bucket, Key=str(key))
            return True
        except ClientError:
            response = self.client.list_objects_v2(Bucket=self.config.bucket, Prefix=str(key), MaxKeys=1)
            return "Contents" in response

    def list_keys(self, prefix: StorageKey | None = None) -> list[StorageKey]:
        """List keys with optional prefix."""
        self._ensure_bucket()

        try:
            paginator = self.client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.config.bucket, Prefix=str(prefix) if prefix else "")

            keys: list[StorageKey] = []
            for page in pages:
                if "Contents" in page:
                    keys.extend(StorageKey(obj["Key"]) for obj in page["Contents"] if "Key" in obj)

            return keys
        except ClientError as e:
            raise S3StorageError("Failed to list keys") from e

    def get_uri(self, key: StorageKey) -> str:
        """Get S3 URI for key."""
        return f"s3://{self.config.bucket}/{key}"

    @classmethod
    def from_settings(cls, bucket: str | None = None) -> S3Storage:
        if bucket is None:
            raise ValueError("Bucket name is required")
        config = S3Config(bucket=bucket)
        return cls(config)
