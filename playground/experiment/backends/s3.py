from __future__ import annotations

import asyncio
import json
import platform
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Self

from fatum.experiment.types import StorageKey
from fatum.reproducibility.git import GitInfo

if TYPE_CHECKING:
    import aioboto3
    from mypy_boto3_s3.client import S3Client
    from mypy_boto3_s3.paginator import ListObjectsV2Paginator
    from types_aiobotocore_s3.client import S3Client as AsyncS3Client


class S3Config(BaseModel):
    """Comprehensive S3 configuration with all boto3 and transfer options.

    Examples
    --------
    >>> # MinIO configuration
    >>> config = S3Config(
    ...     bucket="experiments",
    ...     endpoint_url="http://localhost:9000",
    ...     aws_access_key_id="minioadmin",
    ...     aws_secret_access_key="minioadmin",
    ...     addressing_style="path",
    ...     use_ssl=False
    ... )

    >>> # AWS S3 with encryption
    >>> config = S3Config(
    ...     bucket="my-bucket",
    ...     region_name="us-west-2",
    ...     server_side_encryption="AES256"
    ... )

    >>> # High performance configuration
    >>> config = S3Config(
    ...     bucket="data",
    ...     max_pool_connections=50,
    ...     max_concurrency=20,
    ...     multipart_threshold_mb=25
    ... )
    """

    model_config = ConfigDict(extra="ignore")

    bucket: str = Field(description="S3 bucket name")

    endpoint_url: str | None = Field(default=None, description="S3 endpoint URL")
    region_name: str = Field(default="us-east-1", description="AWS region")
    aws_access_key_id: str | None = Field(default=None, description="AWS access key")
    aws_secret_access_key: str | None = Field(default=None, description="AWS secret key")
    aws_session_token: str | None = Field(default=None, description="AWS session token")

    signature_version: str = Field(default="s3v4", description="Signature version")
    addressing_style: Literal["path", "virtual", "auto"] = Field(default="auto", description="S3 addressing style")
    use_ssl: bool = Field(default=True, description="Use SSL/TLS")
    verify: bool | str = Field(default=True, description="SSL certificate verification")

    server_side_encryption: Literal["AES256", "aws:kms"] | None = Field(
        default=None, description="Server-side encryption method"
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

    experiments_prefix: str = Field(default="experiments", description="Experiments S3 prefix")
    metadata_dir: str = Field(default="metadata", description="Metadata subdirectory")
    artifacts_dir: str = Field(default="artifacts", description="Artifacts subdirectory")
    documents_dir: str = Field(default="documents", description="Documents subdirectory")

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


class SystemInfo(BaseModel):
    """System information for experiment tracking."""

    platform: str = Field(default_factory=lambda: platform.system())
    platform_release: str = Field(default_factory=lambda: platform.release())
    platform_version: str = Field(default_factory=lambda: platform.version())
    architecture: str = Field(default_factory=lambda: platform.machine())
    python_version: str = Field(default_factory=lambda: platform.python_version())
    node: str = Field(default_factory=lambda: platform.node())


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
    """S3 storage with comprehensive configuration support.

    This class consolidates functionality from DocumentS3Storage and provides
    a clean, type-safe interface for S3 operations with full configuration control.

    Examples
    --------
    >>> # Using configuration object
    >>> config = S3Config(bucket="my-bucket", region_name="us-west-2")
    >>> storage = S3Storage(config)

    >>> # Using factory for MinIO
    >>> storage = S3Storage.for_minio(bucket="experiments")

    >>> # Using factory for AWS
    >>> storage = S3Storage.for_aws(bucket="data", region_name="eu-west-1")

    >>> # With injected client for testing
    >>> storage = S3Storage(config, client=mock_client)
    """

    def __init__(
        self,
        config: S3Config,
        client: S3Client | None = None,
    ) -> None:
        """Initialize S3 storage with configuration.

        Parameters
        ----------
        config : S3Config
            Comprehensive S3 configuration
        client : S3Client | None
            Optional S3 client to inject (for testing)
        """
        self.config = config
        self.cache_dir = config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = client
        self._owns_client = client is None
        if self._client is None:
            self._client = self._create_client()

        self._transfer_config = config.get_transfer_config()

        self._run_id: str | None = None
        self._experiment_id: str | None = None

        self._bucket_verified = False
        if config.auto_create_bucket:
            self._ensure_bucket()

    def _create_client(self) -> S3Client:
        """Create S3 client from configuration."""
        try:
            boto_config = self.config.get_boto_config()
            client_kwargs = self.config.get_client_kwargs()
            return boto3.client("s3", config=boto_config, **client_kwargs)  # type: ignore[return-value]
        except NoCredentialsError as e:
            raise S3ConnectionError("No AWS credentials found", {"error": str(e)}) from e
        except Exception as e:
            raise S3ConnectionError("Failed to create S3 client", {"error": str(e)}) from e

    @property
    def client(self) -> S3Client:
        """Get S3 client."""
        assert self._client is not None
        return self._client

    def _ensure_bucket(self) -> None:
        """Ensure bucket exists."""
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
        """Initialize experiment run tracking.

        Parameters
        ----------
        run_id : str
            Unique run identifier
        experiment_id : str
            Experiment identifier
        """
        self._run_id = run_id
        self._experiment_id = experiment_id

        metadata = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "started_at": datetime.now().isoformat(),
            "git_info": GitInfo.current().model_dump() if GitInfo.current() else None,
            "system_info": SystemInfo().model_dump(),
        }
        self._save_json(f"{self.config.metadata_dir}/run_init.json", metadata)

    def finalize(self, status: str) -> None:
        """Finalize experiment run.

        Parameters
        ----------
        status : str
            Final status of the run
        """
        final_metadata = {
            "status": status,
            "ended_at": datetime.now().isoformat(),
        }
        self._save_json(f"{self.config.metadata_dir}/run_final.json", final_metadata)

    def save_documents(self, document_dir: Path, prefix: str | None = None) -> dict[str, str]:
        """Save documents directory to S3 with concurrent uploads.

        Parameters
        ----------
        document_dir : Path
            Directory containing documents to upload
        prefix : str | None
            S3 prefix for documents (default: config.documents_dir)

        Returns
        -------
        dict[str, str]
            Mapping of local paths to S3 keys for uploaded files
        """
        if prefix is None:
            prefix = self.config.documents_dir

        if not document_dir.exists():
            raise FileNotFoundError(f"Document directory not found: {document_dir}")

        file_pairs: list[tuple[Path, str]] = []
        base_key = self._build_key(prefix)

        if document_dir.is_dir():
            for file_path in document_dir.rglob("*"):
                if file_path.is_file():
                    relative = file_path.relative_to(document_dir)
                    s3_key = f"{base_key}/{relative}".replace("\\", "/")
                    file_pairs.append((file_path, s3_key))
        else:
            s3_key = f"{base_key}/{document_dir.name}"
            file_pairs.append((document_dir, s3_key))

        if not file_pairs:
            return {}

        results = {}
        extra_args = self.config.get_extra_args()

        def upload_single(local_path: Path, key: str) -> tuple[str, str, bool]:
            try:
                file_size = local_path.stat().st_size
                if file_size > self._transfer_config.multipart_threshold:
                    self.client.upload_file(
                        str(local_path),
                        self.config.bucket,
                        key,
                        Config=self._transfer_config,
                        ExtraArgs=extra_args if extra_args else None,
                    )
                else:
                    self.client.upload_file(
                        str(local_path),
                        self.config.bucket,
                        key,
                        ExtraArgs=extra_args if extra_args else None,
                    )
                return str(local_path), key, True
            except Exception:
                return str(local_path), key, False

        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as executor:
            futures = [executor.submit(upload_single, path, key) for path, key in file_pairs]
            for future in as_completed(futures):
                local_path, s3_key, success = future.result()
                if success:
                    results[local_path] = s3_key

        if results:
            stats = {
                "count": len(results),
                "total_size": sum(Path(p).stat().st_size for p in results),
                "uploaded_at": datetime.now().isoformat(),
            }
            self._save_json(f"{self.config.metadata_dir}/document_stats.json", stats)

        return results

    def save_config(self, config: dict[str, Any]) -> None:
        """Save configuration to S3.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary to save
        """
        self._save_json("config.json", config)

    def save_artifact(self, key: str, source: Path) -> str:
        """Save artifact to S3.

        Parameters
        ----------
        key : str
            Artifact key/name
        source : Path
            Source file path

        Returns
        -------
        str
            S3 URI of saved artifact, empty string on failure
        """
        s3_key = self._build_key(f"{self.config.artifacts_dir}/{key}")
        extra_args = self.config.get_extra_args()

        try:
            file_size = source.stat().st_size
            if file_size > self._transfer_config.multipart_threshold:
                self.client.upload_file(
                    str(source),
                    self.config.bucket,
                    s3_key,
                    Config=self._transfer_config,
                    ExtraArgs=extra_args if extra_args else None,
                )
            else:
                self.client.upload_file(
                    str(source),
                    self.config.bucket,
                    s3_key,
                    ExtraArgs=extra_args if extra_args else None,
                )
            return f"s3://{self.config.bucket}/{s3_key}"
        except Exception:
            return ""

    def download_directory(self, prefix: str | None = None, local_dir: Path | None = None) -> dict[str, Path]:
        """Download directory from S3 with concurrent downloads.

        Parameters
        ----------
        prefix : str | None
            S3 prefix to download from
        local_dir : Path | None
            Local directory to download to

        Returns
        -------
        dict[str, Path]
            Mapping of S3 keys to local paths for downloaded files
        """
        if prefix is None:
            prefix = self._build_key("")

        if local_dir is None:
            if self._experiment_id and self._run_id:
                local_dir = self.cache_dir / self._experiment_id / self._run_id
            else:
                local_dir = self.cache_dir

        local_dir.mkdir(parents=True, exist_ok=True)

        keys: list[str] = []
        paginator: ListObjectsV2Paginator = self.client.get_paginator("list_objects_v2")  # type: ignore[assignment]

        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
            if "Contents" in page:
                keys.extend(
                    obj["Key"]
                    for obj in page["Contents"]
                    if "Key" in obj and obj["Key"] and not obj["Key"].endswith("/")
                )

        if not keys:
            return {}

        results = {}

        def download_single(key: str) -> tuple[str, Path | None]:
            try:
                local_path = local_dir / key[len(prefix) :].lstrip("/")
                local_path.parent.mkdir(parents=True, exist_ok=True)

                response = self.client.head_object(Bucket=self.config.bucket, Key=key)
                file_size = response.get("ContentLength", 0)

                if file_size > self._transfer_config.multipart_threshold:
                    self.client.download_file(self.config.bucket, key, str(local_path), Config=self._transfer_config)
                else:
                    self.client.download_file(self.config.bucket, key, str(local_path))

                return key, local_path
            except Exception:
                return key, None

        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as executor:
            futures = [executor.submit(download_single, key) for key in keys]
            for future in as_completed(futures):
                key, path = future.result()
                if path:
                    results[key] = path

        return results

    def _build_key(self, suffix: str = "") -> str:
        """Build S3 key with experiment prefix."""
        if self._experiment_id and self._run_id:
            base = f"{self.config.experiments_prefix}/{self._experiment_id}/runs/{self._run_id}"
            return f"{base}/{suffix}" if suffix else base
        return suffix

    def _save_json(self, relative_path: str, data: dict[str, Any]) -> None:
        """Save JSON data to S3."""
        if not self._run_id or not self._experiment_id:
            return

        key = self._build_key(relative_path)
        temp_file = Path(tempfile.mktemp(suffix=".json"))
        extra_args = self.config.get_extra_args()

        try:
            temp_file.write_text(json.dumps(data, indent=2, default=str))
            self.client.upload_file(
                str(temp_file), self.config.bucket, key, ExtraArgs=extra_args if extra_args else None
            )
        finally:
            temp_file.unlink(missing_ok=True)

    def save(self, key: StorageKey, source: Path, **kwargs: Any) -> None:
        """Save file or directory to S3.

        Parameters
        ----------
        key : StorageKey
            S3 key
        source : Path
            Source file or directory
        **kwargs : Any
            Additional arguments passed to upload_file()
        """
        self._ensure_bucket()

        if not source.exists():
            raise S3StorageError(f"Source does not exist: {source}")

        extra_args = {**self.config.get_extra_args(), **kwargs.get("ExtraArgs", {})}

        try:
            if source.is_file():
                self.client.upload_file(
                    str(source), self.config.bucket, str(key), ExtraArgs=extra_args if extra_args else None
                )
            else:
                # Upload directory
                for file_path in source.rglob("*"):
                    if file_path.is_file():
                        relative = file_path.relative_to(source)
                        s3_key = f"{key}/{relative}".replace("\\", "/")
                        self.client.upload_file(
                            str(file_path),
                            self.config.bucket,
                            s3_key,
                            ExtraArgs=extra_args if extra_args else None,
                        )
        except ClientError as e:
            raise S3StorageError(f"Failed to upload: {key}") from e

    def load(self, key: StorageKey, **kwargs: Any) -> Path:
        """Load file or directory from S3.

        Parameters
        ----------
        key : StorageKey
            S3 key
        **kwargs : Any
            Additional arguments passed to download_file()

        Returns
        -------
        Path
            Local path to downloaded content
        """
        self._ensure_bucket()

        local_path = self.cache_dir / str(key)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            response = self.client.list_objects_v2(Bucket=self.config.bucket, Prefix=str(key))

            if "Contents" not in response:
                raise S3ObjectNotFoundError(f"Key not found: {key}")

            objects = response["Contents"]

            if len(objects) == 1 and objects[0].get("Key") == str(key):
                self.client.download_file(self.config.bucket, str(key), str(local_path), **kwargs)
            else:
                for obj in objects:
                    if "Key" in obj:
                        obj_key = obj["Key"]
                        relative = obj_key[len(str(key)) :].lstrip("/")
                        if relative:
                            file_path = local_path / relative
                            file_path.parent.mkdir(parents=True, exist_ok=True)
                            self.client.download_file(self.config.bucket, obj_key, str(file_path), **kwargs)

            return local_path

        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                raise S3ObjectNotFoundError(f"Key not found: {key}") from e
            raise S3StorageError(f"Failed to download: {key}") from e

    def exists(self, key: StorageKey) -> bool:
        """Check if key exists in S3."""
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
            paginator: ListObjectsV2Paginator = self.client.get_paginator("list_objects_v2")  # type: ignore[assignment]
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
    def for_minio(
        cls,
        bucket: str = "experiments",
        endpoint_url: str = "http://localhost:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        **overrides: Any,
    ) -> S3Storage:
        """Create storage configured for MinIO.

        Parameters
        ----------
        bucket : str
            Bucket name
        endpoint_url : str
            MinIO endpoint URL
        access_key : str
            MinIO access key
        secret_key : str
            MinIO secret key
        **overrides : Any
            Additional configuration overrides

        Returns
        -------
        S3Storage
            Configured MinIO storage
        """
        config = S3Config(
            bucket=bucket,
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            addressing_style="path",
            use_ssl=endpoint_url.startswith("https"),
            verify=False,
            **overrides,
        )
        return cls(config)

    @classmethod
    def for_aws(
        cls,
        bucket: str,
        region_name: str = "us-east-1",
        **overrides: Any,
    ) -> S3Storage:
        """Create storage configured for AWS S3.

        Parameters
        ----------
        bucket : str
            S3 bucket name
        region_name : str
            AWS region
        **overrides : Any
            Additional configuration overrides

        Returns
        -------
        S3Storage
            Configured AWS S3 storage
        """
        config = S3Config(
            bucket=bucket,
            region_name=region_name,
            addressing_style="virtual",
            **overrides,
        )
        return cls(config)

    @classmethod
    def from_env(cls, bucket: str | None = None) -> S3Storage:
        """Create storage from environment variables.

        Environment variables are prefixed with S3_
        (e.g., S3_BUCKET, S3_REGION_NAME, S3_AWS_ACCESS_KEY_ID)

        Parameters
        ----------
        bucket : str | None
            Override bucket name

        Returns
        -------
        S3Storage
            Configured storage from environment
        """
        config = S3Config(bucket=bucket or "default")  # type: ignore[arg-type]
        return cls(config)


class AsyncS3Storage:
    """Asynchronous S3 storage implementation.

    Examples
    --------
    >>> # Using configuration
    >>> config = S3Config(bucket="my-bucket", region_name="us-west-2")
    >>> async with AsyncS3Storage(config) as storage:
    ...     await storage.asave(key, source)

    >>> # With injected client
    >>> async with aioboto3.Session().client('s3', **kwargs) as client:
    ...     storage = AsyncS3Storage(config, client=client)
    ...     await storage.asave(key, source)
    """

    def __init__(
        self,
        config: S3Config,
        client: AsyncS3Client | None = None,
    ) -> None:
        """Initialize async S3 storage.

        Parameters
        ----------
        config : S3Config
            S3 configuration
        client : AsyncS3Client | None
            Optional async S3 client
        """
        self.config = config
        self.cache_dir = config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = client
        self._owns_client = client is None
        self._session: aioboto3.Session | None = None
        self._bucket_verified = False

        self._transfer_config = config.get_transfer_config()

    async def _get_client(self) -> AsyncS3Client:
        """Get or create async client."""
        if self._client is None:
            import aioboto3

            if self._session is None:
                self._session = aioboto3.Session()

            boto_config = self.config.get_boto_config()
            client_kwargs = self.config.get_client_kwargs()
            self._client = await self._session.client("s3", config=boto_config, **client_kwargs).__aenter__()  # type: ignore[assignment]

        return self._client

    async def cleanup(self) -> None:
        """Cleanup if we own the client."""
        if self._owns_client and self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None
            self._session = None

    async def __aenter__(self) -> Self:
        """Context manager entry."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Context manager exit."""
        await self.cleanup()

    async def _ensure_bucket(self) -> None:
        """Ensure bucket exists."""
        if self._bucket_verified:
            return

        client = await self._get_client()

        try:
            await client.head_bucket(Bucket=self.config.bucket)
            self._bucket_verified = True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                if self._owns_client and self.config.auto_create_bucket:
                    try:
                        await client.create_bucket(Bucket=self.config.bucket)
                        self._bucket_verified = True
                    except ClientError as create_error:
                        raise S3BucketError(f"Failed to create bucket '{self.config.bucket}'") from create_error
                else:
                    raise S3BucketError(f"Bucket '{self.config.bucket}' does not exist") from e
            else:
                raise S3ConnectionError(f"Failed to access bucket '{self.config.bucket}'") from e

    async def asave(self, key: StorageKey, source: Path, **kwargs: Any) -> None:
        """Save to S3 asynchronously."""
        await self._ensure_bucket()
        client = await self._get_client()

        if not source.exists():
            raise S3StorageError(f"Source does not exist: {source}")

        extra_args = {**self.config.get_extra_args(), **kwargs.get("ExtraArgs", {})}

        try:
            if source.is_file():
                await client.upload_file(
                    str(source), self.config.bucket, str(key), ExtraArgs=extra_args if extra_args else None
                )
            else:
                tasks = []
                for file_path in source.rglob("*"):
                    if file_path.is_file():
                        relative = file_path.relative_to(source)
                        s3_key = f"{key}/{relative}".replace("\\", "/")
                        tasks.append(
                            client.upload_file(
                                str(file_path),
                                self.config.bucket,
                                s3_key,
                                ExtraArgs=extra_args if extra_args else None,
                            )
                        )

                if tasks:
                    sem = asyncio.Semaphore(self.config.max_concurrency)

                    async def upload_with_limit(coro: Any) -> None:
                        async with sem:
                            await coro

                    await asyncio.gather(*[upload_with_limit(t) for t in tasks])

        except ClientError as e:
            raise S3StorageError(f"Failed to upload: {key}") from e

    async def aload(self, key: StorageKey, **kwargs: Any) -> Path:
        """Load from S3 asynchronously."""
        await self._ensure_bucket()
        client = await self._get_client()

        local_path = self.cache_dir / str(key)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            response = await client.list_objects_v2(Bucket=self.config.bucket, Prefix=str(key))

            if "Contents" not in response:
                raise S3ObjectNotFoundError(f"Key not found: {key}")

            objects = response["Contents"]

            if len(objects) == 1 and objects[0].get("Key") == str(key):
                await client.download_file(self.config.bucket, str(key), str(local_path), **kwargs)
            else:
                tasks = []
                for obj in objects:
                    if "Key" in obj:
                        obj_key = obj["Key"]
                        relative = obj_key[len(str(key)) :].lstrip("/")
                        if relative:
                            file_path = local_path / relative
                            file_path.parent.mkdir(parents=True, exist_ok=True)
                            tasks.append(client.download_file(self.config.bucket, obj_key, str(file_path), **kwargs))

                if tasks:
                    sem = asyncio.Semaphore(self.config.max_concurrency)

                    async def download_with_limit(coro: Any) -> None:
                        async with sem:
                            await coro

                    await asyncio.gather(*[download_with_limit(t) for t in tasks])

            return local_path

        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                raise S3ObjectNotFoundError(f"Key not found: {key}") from e
            raise S3StorageError(f"Failed to download: {key}") from e


def create_minio_storage(
    bucket: str = "experiments",
    cache_dir: Path | None = None,
    **kwargs: Any,
) -> S3Storage:
    """Create MinIO storage with sensible defaults (backward compatibility).

    Parameters
    ----------
    bucket : str
        Bucket name
    cache_dir : Path | None
        Cache directory
    **kwargs : Any
        Override any default parameters

    Returns
    -------
    S3Storage
        Configured MinIO storage
    """
    return S3Storage.for_minio(bucket=bucket, cache_dir=cache_dir, **kwargs)
