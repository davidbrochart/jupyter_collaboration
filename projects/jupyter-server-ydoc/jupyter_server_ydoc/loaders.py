# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Coroutine
from copy import deepcopy
from http import HTTPStatus
from logging import Logger, getLogger
from time import time
from typing import Any

from jupyter_server.services.contents.manager import (
    AsyncContentsManager,
    ContentsManager,
)
from jupyter_server.utils import ensure_async
from jupyter_server_fileid.manager import BaseFileIdManager
from tornado.web import HTTPError

from .utils import OutOfBandChanges


class FileLoader:
    """
    A class to centralize all the operation on a file.
    """

    def __init__(
        self,
        file_id: str,
        file_id_manager: BaseFileIdManager,
        contents_manager: AsyncContentsManager | ContentsManager,
        log: Logger | None = None,
        poll_interval: float | None = None,
        max_consecutive_logs: int = 3,
        stop_poll_on_errors_after: float | None = None,
    ) -> None:
        self._file_id: str = file_id

        self._lock = asyncio.Lock()
        self._poll_interval = poll_interval
        self._stop_poll_on_errors_after = stop_poll_on_errors_after
        self._file_id_manager = file_id_manager
        self._max_consecutive_logs = max_consecutive_logs
        self._contents_manager = contents_manager

        self._log = log or getLogger(__name__)
        self._subscriptions: dict[str, Callable[[], Coroutine[Any, Any, None]]] = {}
        self._filepath_subscriptions: dict[str, Callable[[], Coroutine[Any, Any, None] | None]] = {}

        self._watcher = asyncio.create_task(self._watch_file()) if self._poll_interval else None
        self.last_modified = None
        self._disk_hash: tuple[str | None, str] | None = None
        self._hash_from_content = False
        self._content_format: str | None = None
        self._content_type: str | None = None
        self._failed_save: tuple[str, Any] | None = None
        self._outofband = False
        self.deleted = False
        self._detached_path: str | None = None
        self._current_path = self.path

    @property
    def file_id(self) -> str:
        """File ID"""
        return self._file_id

    @property
    def path(self) -> str:
        """
        The file path.
        """
        if self._detached_path is not None:
            return self._detached_path
        path = self._file_id_manager.get_path(self.file_id)
        if path is None and hasattr(self, "_current_path"):
            # File-ID managers may remove the mapping when the file disappears.
            # Retain its last path so the missing file can still be reported.
            return self._current_path
        if path is None:
            raise RuntimeError(f"No path found for file ID '{self.file_id}'")
        return path

    @property
    def number_of_subscriptions(self) -> int:
        """
        The number of rooms subscribed to this file.
        """
        return len(self._subscriptions)

    async def clean(self) -> None:
        """
        Clean up the file.

        Stops the watch task.
        """
        if self._watcher is not None:
            if not self._watcher.cancelled():
                self._watcher.cancel()
            try:
                await self._watcher
            except asyncio.CancelledError:
                self._log.info(f"File watcher for '{self.file_id}' was cancelled")

    def observe(
        self,
        id: str,
        callback: Callable[[], Coroutine[Any, Any, None]],
        filepath_callback: Callable[[], Coroutine[Any, Any, None] | None] | None = None,
    ) -> None:
        """
        Subscribe to the file to get notified about out-of-band file changes.

            Parameters:
                    id (str): Room ID
                    callback (Callable): Callback for notifying the room.
        """
        self._subscriptions[id] = callback
        if filepath_callback is not None:
            self._filepath_subscriptions[id] = filepath_callback

    def unobserve(self, id: str) -> None:
        """
        Unsubscribe to the file.

            Parameters:
                id (str): Room ID
        """
        del self._subscriptions[id]
        if id in self._filepath_subscriptions.keys():
            del self._filepath_subscriptions[id]

    async def load_content(
        self,
        format: str,
        file_type: str,
        *,
        require_hash: bool = True,
        update_baseline: bool = True,
    ) -> dict[str, Any]:
        """
        Load the content of the file.

            Parameters:
                format (str): File format.
                file_type (str): Content type.
                require_hash (bool): Whether to request the disk content hash.
                update_baseline (bool): Record this disk version immediately.
                    Reloads defer this until the content is successfully applied.

            Returns:
                model (dict): A dictionary with the metadata and content of the file.
        """
        async with self._lock:
            model = await self._get_model_with_hash(
                self.path,
                format=format,
                type=file_type,
                content=True,
                require_hash=require_hash,
            )
            if (
                file_type == "file"
                and "content" in model
                and model["content"]
                and "\r\n" in model["content"]
            ):
                model["content"] = model["content"].replace("\r\n", "\n")
                self._log.debug("Normalizing line endings for %s file on content load", self.path)
            if update_baseline:
                self._record_disk_version(model)
            return model

    async def acknowledge_reload(self, model: dict[str, Any]) -> None:
        """Allow saves after disk content was successfully applied to the room."""
        async with self._lock:
            self._record_disk_version(model)
            self._outofband = False
            self.deleted = False

    async def maybe_save_content(self, model: dict[str, Any]) -> dict[str, Any] | None:
        """
        Save the content of the file.

            Parameters:
                model (dict): A dictionary with format, type, last_modified,
                    and content of the file.

            Raises:
                OutOfBandChanges: if the file was modified at a latter time than the model

        ### Note:
            If there is changes on disk, this method will raise an OutOfBandChanges exception.
        """
        async with self._lock:
            if self._detached_path is not None:
                raise OutOfBandChanges
            path = self.path
            if model["type"] not in {"directory", "file", "notebook"}:
                # fall back to file if unknown type, the content manager only knows
                # how to handle these types
                model["type"] = "file"

            m = await self._get_disk_model(
                path, format=model["format"], type=model["type"], content=False, require_hash=True
            )
            if m is None:
                raise OutOfBandChanges
            # Skip saving if file is not writable
            if not m["writable"]:
                return None

            if not self._outofband and self._disk_version_matches(m):
                self._log.info("Saving file: %s", path)
                # saving is shielded so that it cannot be cancelled
                # otherwise it could corrupt the file
                task = asyncio.create_task(self._save_content(model, path))
                while True:
                    try:
                        return await asyncio.shield(task)
                    except asyncio.CancelledError:
                        if task.cancelled():
                            raise
                        # Repeated autosave cancellations must not release the
                        # lock before the write and hash baseline update finish.
            else:
                # file changed on disk, raise an error
                self._outofband = True
                raise OutOfBandChanges

    async def check_outofband_content(self, model: dict[str, Any]) -> tuple[bool, bool]:
        """Return whether disk changed externally and whether shared edits remain unsaved."""
        async with self._lock:
            if self._detached_path is not None:
                # Preserve the session, but refresh whether its former path exists.
                await self._get_disk_model(self.path, content=False)
                return True, True
            self._outofband = True
            path = self.path
            disk = await self._get_disk_model(
                path, format=model["format"], type=model["type"], content=True, require_hash=True
            )
            if disk is None:
                return True, True
            content = disk["content"]
            if model["type"] == "file" and isinstance(content, str):
                content = content.replace("\r\n", "\n")
            if model["type"] == "notebook":
                # YNotebook adds these defaults on load. Compare the same defaults
                # without changing the disk content or the shared snapshot.
                content = self._notebook_with_metadata_defaults(content)
                model = {
                    **model,
                    "content": self._notebook_with_metadata_defaults(model["content"]),
                }
            unsaved = content != model["content"]
            saved_before_rename = False
            if self._failed_save is not None:
                failed_path, saved_content = self._failed_save
                if model["type"] == "notebook":
                    saved_content = self._notebook_with_metadata_defaults(saved_content)
                elif isinstance(saved_content, str):
                    saved_content = saved_content.replace("\r\n", "\n")
                saved_before_rename = path != failed_path and content == saved_content
            # A rename can briefly make the old path disappear, and a queued
            # poll can also notify after a save already updated our baseline.
            # Compare hashes against the last disk version, not timestamps or
            # the live document: external writes can preserve modification times,
            # while the live document can legitimately contain unsaved edits.
            if self._disk_version_matches(disk) or saved_before_rename or not unsaved:
                self._record_disk_version(disk)
                self._outofband = False
                return False, unsaved
            return True, unsaved

    def _disk_version_matches(self, model: dict[str, Any]) -> bool:
        return self._disk_hash is not None and self._disk_hash == (
            model.get("hash_algorithm"),
            model.get("hash"),
        )

    def _record_disk_version(self, model: dict[str, Any]) -> None:
        self.last_modified = model["last_modified"]
        content_hash = model.get("hash")
        self._disk_hash = (
            (model.get("hash_algorithm"), content_hash) if content_hash is not None else None
        )
        self._failed_save = None

    async def _get_model_with_hash(self, path: str, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("require_hash", True)
        if self._hash_from_content:
            # Keep using the same representation for the whole session, even
            # if the backend later starts returning its own hashes.
            kwargs = {
                **kwargs,
                "content": True,
                "format": self._content_format,
                "type": self._content_type,
            }
        model = await ensure_async(self._contents_manager.get(path, **kwargs))
        if model.get("hash") is not None and not self._hash_from_content:
            return model
        if model.get("content") is None:
            model = await ensure_async(
                self._contents_manager.get(path, **{**kwargs, "content": True})
            )
            if model.get("hash") is not None and not self._hash_from_content:
                return model
        if model.get("content") is None:
            raise ValueError(f"Contents manager returned neither content nor a hash for {path}")
        self._hash_from_content = True
        self._content_format = model.get("format")
        self._content_type = model.get("type")
        # Hash the disk snapshot, never the live shared document. Sorting keys
        # makes JSON object ordering irrelevant; list order remains significant.
        data = json.dumps(
            {key: model.get(key) for key in ("type", "format", "content")},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {**model, "hash": hashlib.sha256(data).hexdigest(), "hash_algorithm": "sha256"}

    @staticmethod
    def _notebook_with_metadata_defaults(content: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(content.get("metadata", {}))
        metadata.setdefault("language_info", {"name": ""})
        metadata.setdefault("kernelspec", {"name": "", "display_name": ""})
        return {**content, "metadata": metadata}

    async def _get_disk_model(self, path: str, **kwargs: Any) -> dict[str, Any] | None:
        """Treat disappearance of a previously loaded file as an external change."""
        try:
            model = await self._get_model_with_hash(path, **kwargs)
        except HTTPError as error:
            if error.status_code != HTTPStatus.NOT_FOUND or self.last_modified is None:
                raise
            self.deleted = True
            self._outofband = True
            return None
        self.deleted = False
        return model

    async def detach(self) -> None:
        """Let new clients open the disk version without changing this session.

        Existing clients retain their room ID and in-memory document. Removing
        its path index gives the disk version a fresh ID on its next open.
        """
        async with self._lock:
            if self._detached_path is None:
                path = self.path
                if self._file_id_manager.get_id(path) == self.file_id:
                    self._file_id_manager.delete(path)
                self._detached_path = path
                self._outofband = True

    async def _save_content(self, model: dict[str, Any], path: str) -> dict[str, Any]:
        try:
            return await self._save_content_at_path(model, path)
        except Exception as save_error:
            # Contents managers can write successfully and then fail while
            # checkpointing a path that was just renamed. Retain that write's
            # snapshot until we can verify it at the new path; it must not be
            # mistaken for an external write when newer shared edits arrive.
            self._failed_save = (path, deepcopy(model["content"]))
            # The file may have been renamed after the metadata check in
            # maybe_save_content but before the save reached the contents
            # manager. Retry only if the stable file ID now resolves to a
            # different path; errors for deleted files must still surface.
            try:
                new_path = self.path
            except Exception:
                # Deleting a file also removes its file ID. Do not replace
                # the original contents manager error with the path lookup
                # failure in that case.
                raise save_error from None
            if new_path == path:
                raise
            self._log.info("File moved while saving: %s -> %s; retrying", path, new_path)
            return await self._save_content_at_path(model, new_path)

    async def _save_content_at_path(self, model: dict[str, Any], path: str) -> dict[str, Any]:
        m = await ensure_async(self._contents_manager.save(model, path))
        # TODO, get rid of the extra `get` here once upstream issue:
        # https://github.com/jupyter-server/jupyter_server/issues/1453 is resolved
        model_with_hash = await self._get_model_with_hash(
            path,
            content=False,
            require_hash=True,
        )
        self._record_disk_version(model_with_hash)
        return {
            **m,
            "hash": model_with_hash["hash"],
            "hash_algorithm": model_with_hash.get("hash_algorithm"),
        }

    async def _watch_file(self) -> None:
        """
        Async task for watching a file.
        """
        self._log.info("Watching file: %s", self.path)

        if self._poll_interval is None:
            return

        consecutive_error_logs = 0
        suppression_logged = False
        consecutive_errors_started = None

        while True:
            try:
                await asyncio.sleep(self._poll_interval)
                try:
                    await self.maybe_notify()
                    consecutive_error_logs = 0
                    suppression_logged = False
                    consecutive_errors_started = None
                except Exception as e:
                    # We do not want to terminate the watcher if the content manager request
                    # fails due to timeout, server error or similar temporary issue; we only
                    # terminate if the file is not found or we get unauthorized error for
                    # an extended period of time.
                    if isinstance(e, HTTPError) and e.status_code in {
                        HTTPStatus.NOT_FOUND,
                        HTTPStatus.UNAUTHORIZED,
                    }:
                        if (
                            consecutive_errors_started
                            and self._stop_poll_on_errors_after is not None
                        ):
                            errors_duration = time() - consecutive_errors_started
                            if errors_duration > self._stop_poll_on_errors_after:
                                self._log.warning(
                                    (
                                        "Stopping watching file due to consecutive "
                                        "errors over %s seconds: %s"
                                    ),
                                    self._stop_poll_on_errors_after,
                                    self.path,
                                )
                                break
                        else:
                            consecutive_errors_started = time()
                    # Otherwise we just log the error
                    if consecutive_error_logs < self._max_consecutive_logs:
                        self._log.error("Error watching file %s: %s", self.path, e, exc_info=e)
                        consecutive_error_logs += 1
                    elif not suppression_logged:
                        self._log.warning(
                            "Too many errors while watching %s - suppressing further logs.",
                            self.path,
                        )
                        suppression_logged = True

            except asyncio.CancelledError:
                break

    async def maybe_notify(self) -> None:
        """
        Notifies subscribed rooms about out-of-band file changes.
        """
        do_notify = False
        filepath_change = False
        async with self._lock:
            if self._detached_path is not None:
                was_deleted = self.deleted
                await self._get_disk_model(self.path, content=False)
                # Detached rooms never reload or detach the new disk session.
                # Only existence changes need a new notification.
                if self.deleted == was_deleted:
                    return
                do_notify = True
            path = self.path
            if self._current_path != path:
                self._current_path = path
                filepath_change = True

            # Hash polling also catches external writes that preserve timestamps.
            model = await self._get_disk_model(path, content=False, require_hash=True)

            if (
                model is None
                or self._outofband
                or (
                    self.last_modified is not None
                    and (
                        self.last_modified != model["last_modified"]
                        or not self._disk_version_matches(model)
                    )
                )
            ):
                do_notify = True
                self._outofband = True
            else:
                self.last_modified = model["last_modified"]

        if do_notify:
            # Notify out-of-band change
            # callbacks will load the file content, thus release the lock before calling them
            for callback in tuple(self._subscriptions.values()):
                await callback()

        if filepath_change:
            # Clear any transient deletion notice before publishing the renamed
            # path. Otherwise a save immediately after the tab title updates
            # can still be blocked by the notice for the old path.
            for filepath_callback in self._filepath_subscriptions.values():
                await ensure_async(filepath_callback())


class FileLoaderMapping:
    """Map rooms to file loaders."""

    def __init__(
        self,
        settings: dict,
        log: Logger | None = None,
        file_poll_interval: float | None = None,
        file_stop_poll_on_errors_after: float | None = None,
    ) -> None:
        """
        Args:
            settings: Server settings
            log: [optional] Server log; default to local logger
            file_poll_interval: [optional] Interval between room
                notification; default the loader won't poll
        """
        self._settings = settings
        self.__dict: dict[str, FileLoader] = {}
        self.log = log or getLogger(__name__)
        self.file_poll_interval = file_poll_interval
        self._stop_poll_on_errors_after = file_stop_poll_on_errors_after

    @property
    def contents_manager(self) -> AsyncContentsManager | ContentsManager:
        return self._settings["contents_manager"]

    @property
    def file_id_manager(self) -> BaseFileIdManager:
        return self._settings["file_id_manager"]

    def __contains__(self, file_id: str) -> bool:
        """Test if a file has a loader."""
        return file_id in self.__dict

    def __getitem__(self, file_id: str) -> FileLoader:
        """Get the loader for a given file.

        If there is none, create one.
        """
        path = self.file_id_manager.get_path(file_id)

        # Instantiate the FileLoader if it doesn't exist yet
        file = self.__dict.get(file_id)
        if file is None:
            self.log.info("Creating FileLoader for: %s", path)
            file = FileLoader(
                file_id,
                self.file_id_manager,
                self.contents_manager,
                self.log,
                self.file_poll_interval,
                stop_poll_on_errors_after=self._stop_poll_on_errors_after,
            )
            self.__dict[file_id] = file

        return file

    async def __delitem__(self, file_id: str) -> None:
        """Delete a loader for a given file."""
        await self.remove(file_id)

    async def clear(self) -> None:
        """Clear all loaders."""
        tasks = []
        for id in list(self.__dict):
            loader = self.__dict.pop(id)
            tasks.append(loader.clean())

        await asyncio.gather(*tasks)

    async def remove(self, file_id: str) -> None:
        """Remove the loader for a given file."""
        loader = self.__dict.pop(file_id)
        await loader.clean()
