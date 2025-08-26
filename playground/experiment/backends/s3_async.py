from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import nest_asyncio
from botocore.exceptions import ClientError
from typing_extensions import Self

from fatum.experiment.types import StorageKey
from playground.experiment.backends.s3 import (
    S3BucketError,
    S3Config,
    S3ConnectionError,
    S3ObjectNotFoundError,
    S3StorageError,
    SystemInfo,
)

if TYPE_CHECKING:
    import aioboto3
    from types_aiobotocore_s3.client import S3Client as AsyncS3Client


class AsyncS3Storage:
    def __init__(
        self,
        config: S3Config,
        client: AsyncS3Client | None = None,
    ) -> None:
        self.config = config
        self.cache_dir = config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = client
        self._owns_client = client is None
        self._session: aioboto3.Session | None = None
        self._bucket_verified = False

        self._transfer_config = config.get_transfer_config()
        self._run_id: str | None = None
        self._experiment_id: str | None = None

    async def _get_client(self) -> AsyncS3Client:
        if self._client is None:
            import aioboto3

            if self._session is None:
                self._session = aioboto3.Session()

            boto_config = self.config.get_boto_config()
            client_kwargs = self.config.get_client_kwargs()
            self._client = await self._session.client("s3", config=boto_config, **client_kwargs).__aenter__()  # type: ignore[call-overload]

        return self._client

    async def cleanup(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None
            self._session = None

    async def __aenter__(self) -> Self:
        if self.config.auto_create_bucket:
            await self._ensure_bucket()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.cleanup()

    async def _ensure_bucket(self) -> None:
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

    async def ainitialize(self, run_id: str, experiment_id: str) -> None:
        self._run_id = run_id
        self._experiment_id = experiment_id

        metadata = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "started_at": datetime.now().isoformat(),
            "status": "running",
            "system": SystemInfo().model_dump(),
        }

        metadata_key = f"{experiment_id}/metadata/{run_id}.json"
        await self.asave(metadata_key, metadata)

    async def afinalize(self, status: str) -> None:
        if not self._run_id or not self._experiment_id:
            return

        metadata_key = f"{self._experiment_id}/metadata/{self._run_id}.json"

        try:
            loaded_metadata = await self.aload(metadata_key, as_type="json")
            if isinstance(loaded_metadata, dict):
                metadata_dict: dict[str, Any] = loaded_metadata

                started_at = metadata_dict.get("started_at")
                duration = None
                if started_at and isinstance(started_at, str):
                    start_time = datetime.fromisoformat(started_at)
                    duration = (datetime.now() - start_time).total_seconds()

                metadata_dict["status"] = status
                metadata_dict["ended_at"] = datetime.now().isoformat()
                metadata_dict["duration_seconds"] = duration

                await self.asave(metadata_key, metadata_dict)
        except S3ObjectNotFoundError:
            metadata = {
                "run_id": self._run_id,
                "experiment_id": self._experiment_id,
                "status": status,
                "ended_at": datetime.now().isoformat(),
            }
            await self.asave(metadata_key, metadata)

    def initialize(self, run_id: str, experiment_id: str) -> None:
        nest_asyncio.apply()

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(self.ainitialize(run_id, experiment_id))

    def finalize(self, status: str) -> None:
        nest_asyncio.apply()

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(self.afinalize(status))

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @property
    def experiment_id(self) -> str | None:
        return self._experiment_id

    async def asave(
        self,
        key: str | StorageKey,
        data: Path | str | bytes | dict[str, Any],
    ) -> str | dict[str, str]:
        await self._ensure_bucket()
        client = await self._get_client()
        key = str(key)

        if isinstance(data, Path):
            if not data.exists():
                raise FileNotFoundError(f"Source path not found: {data}")

            if data.is_dir():
                return await self._upload_directory_async(client, key, data)
            else:
                return await self._upload_file_async(client, key, data)

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
                return await self._upload_file_async(client, key, temp_path)
            finally:
                temp_path.unlink(missing_ok=True)

    async def aload(
        self,
        key: str | StorageKey,
        *,
        target: Path | None = None,
        as_type: Literal["bytes", "str", "json", "auto"] = "auto",
    ) -> Path | dict[str, Path] | bytes | str | dict[str, Any] | list[Any]:
        await self._ensure_bucket()
        client = await self._get_client()
        key = str(key)

        objects = await self._list_objects_async(client, key)

        if len(objects) == 0:
            raise S3ObjectNotFoundError(f"Key not found: {key}")

        is_directory = len(objects) > 1 or (len(objects) == 1 and objects[0]["Key"] != key)

        if is_directory:
            target_dir = target or self.cache_dir / key
            return await self._download_directory_async(client, key, target_dir, objects)

        if target:
            return await self._download_file_async(client, key, target)

        data = await self._download_bytes_async(client, key)

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

    async def aexists(self, key: StorageKey) -> bool:
        await self._ensure_bucket()
        client = await self._get_client()

        try:
            await client.head_object(Bucket=self.config.bucket, Key=str(key))
            return True
        except ClientError:
            response = await client.list_objects_v2(Bucket=self.config.bucket, Prefix=str(key), MaxKeys=1)
            return "Contents" in response

    async def alist_keys(self, prefix: StorageKey | None = None) -> list[StorageKey]:
        await self._ensure_bucket()
        client = await self._get_client()

        try:
            paginator = client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.config.bucket, Prefix=str(prefix) if prefix else "")

            keys: list[StorageKey] = []
            async for page in pages:
                if "Contents" in page:
                    keys.extend(StorageKey(obj["Key"]) for obj in page["Contents"] if "Key" in obj)

            return keys
        except ClientError as e:
            raise S3StorageError("Failed to list keys") from e

    def get_uri(self, key: StorageKey) -> str:
        return f"s3://{self.config.bucket}/{key}"

    async def _upload_file_async(self, client: AsyncS3Client, key: str, path: Path) -> str:
        extra_args = self.config.get_extra_args()
        await client.upload_file(str(path), self.config.bucket, key, ExtraArgs=extra_args if extra_args else None)
        return f"s3://{self.config.bucket}/{key}"

    async def _upload_directory_async(self, client: AsyncS3Client, prefix: str, directory: Path) -> dict[str, str]:
        files = [
            (f, f"{prefix}/{f.relative_to(directory)}".replace("\\", "/"))
            for f in directory.rglob("*")
            if f.is_file() and not any(f.match(p) for p in self.config.exclude_patterns)
        ]

        if not files:
            return {}

        results: dict[str, str] = {}
        sem = asyncio.Semaphore(self.config.max_concurrency)

        async def upload_with_limit(local_path: Path, s3_key: str) -> tuple[str, str]:
            async with sem:
                uri = await self._upload_file_async(client, s3_key, local_path)
                return str(local_path), uri

        tasks = [upload_with_limit(local_path, s3_key) for local_path, s3_key in files]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for result in completed:
            if isinstance(result, tuple):
                local_path, uri = result
                results[local_path] = uri
            elif isinstance(result, Exception):
                print(f"Upload failed: {result}")

        return results

    async def _download_file_async(self, client: AsyncS3Client, key: str, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)

        response = await client.head_object(Bucket=self.config.bucket, Key=key)
        size = response.get("ContentLength", 0)

        if size > self._transfer_config.multipart_threshold:
            await client.download_file(self.config.bucket, key, str(target))
        else:
            obj = await client.get_object(Bucket=self.config.bucket, Key=key)
            body = obj["Body"]
            data = await body.read()
            target.write_bytes(data)

        return target

    async def _download_bytes_async(self, client: AsyncS3Client, key: str) -> bytes:
        obj = await client.get_object(Bucket=self.config.bucket, Key=key)
        body = obj["Body"]
        return await body.read()

    async def _download_directory_async(
        self, client: AsyncS3Client, prefix: str, target_dir: Path, objects: list[Any]
    ) -> dict[str, Path]:
        target_dir.mkdir(parents=True, exist_ok=True)

        downloads = []
        for obj in objects:
            s3_key = obj["Key"]
            relative = s3_key[len(str(prefix)) :].lstrip("/")
            if relative:
                local_path = target_dir / relative
                downloads.append((s3_key, local_path))

        results: dict[str, Path] = {}
        sem = asyncio.Semaphore(self.config.max_concurrency)

        async def download_with_limit(s3_key: str, local_path: Path) -> tuple[str, Path]:
            async with sem:
                path = await self._download_file_async(client, s3_key, local_path)
                return s3_key, path

        tasks = [download_with_limit(s3_key, local_path) for s3_key, local_path in downloads]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for result in completed:
            if isinstance(result, tuple):
                s3_key, path = result
                results[s3_key] = path
            elif isinstance(result, Exception):
                print(f"Download failed: {result}")

        return results

    async def _list_objects_async(self, client: AsyncS3Client, prefix: str) -> list[Any]:
        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.config.bucket, Prefix=prefix)

        objects = []
        async for page in pages:
            if "Contents" in page:
                objects.extend(page["Contents"])

        return objects

    @classmethod
    def from_settings(cls, bucket: str | None = None) -> AsyncS3Storage:
        if bucket is None:
            raise ValueError("Bucket name is required")
        config = S3Config(bucket=bucket)
        return cls(config)
