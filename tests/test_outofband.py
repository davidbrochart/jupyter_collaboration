# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import nbformat
import pytest
from jupyter_server.services.contents.filemanager import FileContentsManager
from jupyter_server_ydoc.loaders import FileLoader
from jupyter_server_ydoc.rooms import DocumentRoom
from jupyter_server_ydoc.test_utils import FakeEventLogger
from jupyter_server_ydoc.utils import MessageType, OutOfBandChanges
from pycrdt import Decoder
from pycrdt.websocket import YRoom


@pytest.fixture
def behavior_on_external_change():
    return "ask"


@pytest.fixture
def document(
    jp_root_dir,
    arbitrary_fid_manager,
    request,
    jp_asyncio_loop,
    behavior_on_external_change,
):
    async def create():
        notebook = getattr(request, "param", False)
        path = "document.ipynb" if notebook else "document.txt"
        content = (
            nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("original")])
            if notebook
            else "original"
        )
        # Keep disk operations on the event-loop thread: pycrdt subscriptions
        # and events must not be garbage-collected by a file I/O worker thread.
        cm = FileContentsManager(root_dir=str(jp_root_dir))
        model = {
            "type": "notebook" if notebook else "file",
            "format": "json" if notebook else "text",
            "content": content,
        }
        cm.save(model, path)
        file_id = arbitrary_fid_manager.index(path)
        loader = FileLoader(file_id, arbitrary_fid_manager, cm)
        room = DocumentRoom(
            "test-room",
            model["format"],
            model["type"],
            loader,
            FakeEventLogger(),
            None,
            None,
            behavior_on_external_change=behavior_on_external_change,
        )
        await room.initialize()
        room._document.dirty = False
        return cm, loader, room, model, path

    result = jp_asyncio_loop.run_until_complete(create())
    yield result
    jp_asyncio_loop.run_until_complete(result[2].stop())
    jp_asyncio_loop.run_until_complete(result[1].clean())


@pytest.mark.parametrize("document", [False, True], indirect=True)
@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("trigger", ["poll", "save", "both"])
async def test_external_changes_wait_for_user_without_creating_files(
    document, dirty, trigger, arbitrary_fid_manager
):
    cm, loader, room, model, path = document
    if dirty:
        if model["type"] == "file":
            room._document.source = "unsaved shared edits"
        else:
            content = await room._document.aget()
            content["cells"][0]["source"] = "unsaved shared edits"
            await room._document.aset(content)
    shared = await room._document.aget()
    external = "external version" if model["type"] == "file" else nbformat.v4.new_notebook()
    cm.save({**model, "content": external}, path)
    if trigger == "poll":
        await loader.maybe_notify()
    elif trigger == "save":
        await room._maybe_save_document(None, save_now=True)
    else:
        await asyncio.gather(loader.maybe_notify(), room._maybe_save_document(None, save_now=True))
    assert loader.path == path
    assert await room._document.aget() == shared
    assert cm.get(path)["content"] == external
    assert len(cm.get("")["content"]) == 1
    assert room._document.dirty
    assert room.outofband == {"originalPath": path}
    assert arbitrary_fid_manager.index(path) != loader.file_id
    for _ in range(2):
        await loader.maybe_notify()
        with pytest.raises(OutOfBandChanges):
            await loader.maybe_save_content(model)
    assert len(cm.get("")["content"]) == 1


async def test_identical_external_write_does_not_prompt(document):
    cm, loader, room, model, path = document
    cm.save(model, path)
    await loader.maybe_notify()
    assert loader.path == path
    assert room.outofband is None
    assert not room._document.dirty


@pytest.mark.parametrize("behavior_on_external_change", ["ask", "overwrite"])
@pytest.mark.parametrize("document", [False, True], indirect=True)
@pytest.mark.parametrize("trigger", ["poll", "save", "check"])
async def test_external_write_with_preserved_timestamp(
    document, jp_root_dir, behavior_on_external_change, trigger
):
    cm, loader, room, model, path = document
    shared = await room._document.aget()
    if model["type"] == "file":
        shared = "unsaved shared edits"
        external = "external version"
    else:
        shared["cells"][0]["source"] = "unsaved shared edits"
        external = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("external version")])
    await room._document.aset(shared)
    room._document.dirty = True
    shared = await room._document.aget()
    file = jp_root_dir / path
    stat = file.stat()
    cm.save({**model, "content": external}, path)
    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert cm.get(path, content=False)["last_modified"] == loader.last_modified

    if trigger == "poll":
        await loader.maybe_notify()
    elif trigger == "save":
        await room._maybe_save_document(None, save_now=True)
    else:
        await room._on_outofband_change()

    # A save must never overwrite the external version just because its
    # timestamp still matches the last load.
    disk = cm.get(path)["content"]
    assert (disk if model["type"] == "file" else disk["cells"][0]["source"]) == "external version"
    actual = await room._document.aget()
    if behavior_on_external_change == "ask":
        assert actual == shared
        assert room.outofband == {"originalPath": path}
        assert room._document.dirty
    else:
        assert (actual if model["type"] == "file" else actual["cells"][0]["source"]) == (
            "external version"
        )
        assert room.outofband is None
        assert not room._document.dirty


async def test_successful_save_updates_disk_baseline(document):
    cm, _loader, room, _model, path = document
    room._document.source = "first edit"
    await room._maybe_save_document(None, save_now=True)
    assert cm.get(path)["content"] == "first edit"

    room._document.source = "second, unsaved edit"
    room._document.dirty = True
    await room._on_outofband_change()
    assert room._document.source == "second, unsaved edit"
    assert room.outofband is None
    assert room._document.dirty
    assert cm.get(path)["content"] == "first edit"


@pytest.mark.parametrize("document", [False, True], indirect=True)
async def test_identical_disk_rewrite_preserves_unsaved_edits(document, jp_root_dir):
    _cm, loader, room, model, path = document
    shared = await room._document.aget()
    if model["type"] == "file":
        shared = "unsaved shared edits"
    else:
        shared["cells"][0]["source"] = "unsaved shared edits"
    await room._document.aset(shared)
    room._document.dirty = True
    shared = await room._document.aget()
    file = jp_root_dir / path
    data = file.read_bytes()
    stat = file.stat()
    file.write_bytes(data)
    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))

    await loader.maybe_notify()

    assert await room._document.aget() == shared
    assert room.outofband is None
    assert room._document.dirty
    assert file.read_bytes() == data


@pytest.mark.parametrize("behavior_on_external_change", ["overwrite"])
@pytest.mark.parametrize("document", [False, True], indirect=True)
@pytest.mark.parametrize("external_write", [False, True])
async def test_rename_after_partial_save_preserves_new_edits(
    document, arbitrary_fid_manager, behavior_on_external_change, external_write
):
    cm, loader, room, model, path = document
    first = await room._document.aget()
    if model["type"] == "file":
        first = "first edit"
        second = "second edit"
    else:
        first["cells"][0]["source"] = "first edit"
        second = {**first, "cells": [{**first["cells"][0], "source": "second edit"}]}
    new_path = "renamed-" + path
    save = cm.save

    def save_then_rename(content, save_path):
        save(content, save_path)
        cm.rename_file(path, new_path)
        raise OSError("Checkpoint failed after the file moved")

    with patch.object(cm, "save", side_effect=save_then_rename):
        with pytest.raises(OSError, match="Checkpoint failed"):
            await loader.maybe_save_content({**model, "content": first})
    arbitrary_fid_manager.move(path, new_path)
    if external_write:
        external = (
            "external version"
            if model["type"] == "file"
            else nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("external version")])
        )
        save({**model, "content": external}, new_path)
    await room._document.aset(second)
    room._document.dirty = True
    second = await room._document.aget()

    with patch.object(room._document, "aset", wraps=room._document.aset) as apply:
        await loader.maybe_notify()
        if external_write:
            apply.assert_awaited_once()
        else:
            apply.assert_not_called()
    if external_write:
        actual = await room._document.aget()
        assert (actual if model["type"] == "file" else actual["cells"][0]["source"]) == (
            "external version"
        )
        assert room.outofband is None
        assert not room._document.dirty
        return
    assert await room._document.aget() == second
    assert room.outofband is None
    assert room._document.dirty
    await room._maybe_save_document(None, save_now=True)
    disk = cm.get(new_path)["content"]
    assert (disk if model["type"] == "file" else disk["cells"][0]["source"]) == "second edit"


@pytest.mark.parametrize("behavior_on_external_change", ["overwrite"])
@pytest.mark.parametrize("document", [False, True], indirect=True)
@pytest.mark.parametrize("trigger", ["poll", "save"])
@pytest.mark.parametrize("external_write", [False, True])
async def test_rename_gap_preserves_unsaved_changes(
    document, arbitrary_fid_manager, behavior_on_external_change, trigger, external_write
):
    cm, loader, room, model, path = document
    shared = await room._document.aget()
    if model["type"] == "file":
        shared = "unsaved shared edits"
    else:
        shared["cells"][0]["source"] = "unsaved shared edits"
    await room._document.aset(shared)
    room._document.dirty = True
    shared = await room._document.aget()

    # Reproduce the interval between the filesystem move and the file-ID
    # manager receiving the rename event, without relying on polling delays.
    new_path = "renamed-" + path
    cm.rename_file(path, new_path)
    if trigger == "poll":
        await loader.maybe_notify()
    else:
        await room._maybe_save_document(None, save_now=True)
    assert room.outofband == {"originalPath": path, "reason": "deleted"}

    if external_write:
        external = (
            "external version"
            if model["type"] == "file"
            else nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("external version")])
        )
        cm.save({**model, "content": external}, new_path)
    disk = cm.get(new_path)["content"]
    arbitrary_fid_manager.move(path, new_path)
    notices_at_path_change = []
    loader.observe(
        "rename-observer", AsyncMock(), lambda: notices_at_path_change.append(room.outofband)
    )
    with patch.object(room._document, "aset", wraps=room._document.aset) as apply:
        await loader.maybe_notify()
        # A notification queued before recovery must not reinterpret pending
        # edits as external changes after the transient deletion was cleared.
        await room._on_outofband_change()
        if external_write:
            apply.assert_awaited_once()
        else:
            apply.assert_not_called()

    assert room.outofband is None
    assert notices_at_path_change == [None]
    assert not loader.deleted
    actual = await room._document.aget()
    if external_write and model["type"] == "notebook":
        # YNotebook fills in optional notebook metadata when loading disk content.
        assert actual["cells"] == disk["cells"]
    else:
        assert actual == (disk if external_write else shared)
    assert room._document.dirty is (not external_write)
    assert arbitrary_fid_manager.get_id(new_path) == loader.file_id

    # A rename must leave pending edits saveable at the new path.
    await room._maybe_save_document(None, save_now=True)
    saved = cm.get(new_path)["content"]
    if not external_write:
        assert (saved if model["type"] == "file" else saved["cells"][0]["source"]) == (
            "unsaved shared edits"
        )
    assert not room._document.dirty


@pytest.mark.parametrize("behavior_on_external_change", ["overwrite"])
async def test_rename_gap_resumes_autosave(
    document, arbitrary_fid_manager, behavior_on_external_change
):
    cm, loader, room, _model, path = document
    room._document.source = "unsaved shared edits"
    room._document.dirty = True
    new_path = "renamed-" + path
    cm.rename_file(path, new_path)
    await loader.maybe_notify()
    assert room.outofband == {"originalPath": path, "reason": "deleted"}

    room._save_delay = 0
    arbitrary_fid_manager.move(path, new_path)
    await loader.maybe_notify()

    assert room._saving_document is not None
    await asyncio.wait_for(room._saving_document, timeout=5)
    assert cm.get(new_path)["content"] == "unsaved shared edits"
    assert not room._document.dirty
    assert room.outofband is None


async def test_manual_save_waits_for_update_lock(document):
    cm, _loader, room, _model, path = document
    room._document.source = "unsaved shared edits"
    room._document.dirty = True
    async with room._update_lock:
        save_task = room._save_to_disc()
        assert save_task is not None
    await asyncio.wait_for(save_task, timeout=5)
    assert cm.get(path)["content"] == "unsaved shared edits"
    assert not room._document.dirty


@pytest.mark.parametrize("document", [True], indirect=True)
async def test_touch_notebook_without_optional_metadata_does_not_prompt(
    document, jp_root_dir, arbitrary_fid_manager
):
    cm, loader, room, model, path = document
    disk = cm.get(path)["content"]
    assert "kernelspec" not in disk["metadata"]
    assert "language_info" not in disk["metadata"]
    shared = await room._document.aget()
    file = jp_root_dir / path
    original_bytes = file.read_bytes()
    stat = file.stat()
    # Change only the timestamp, without letting an initial save normalize disk metadata.
    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))

    await loader.maybe_notify()

    assert file.read_bytes() == original_bytes
    assert await room._document.aget() == shared
    assert room.outofband is None
    assert not room._document.dirty
    assert arbitrary_fid_manager.get_id(path) == loader.file_id


@pytest.mark.parametrize("document", [False, True], indirect=True)
async def test_reopening_disk_version_leaves_other_clients_in_old_room(
    document, arbitrary_fid_manager
):
    cm, loader, room, model, path = document
    original = await room._document.aget()
    external = (
        "external"
        if model["type"] == "file"
        else nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("external")])
    )
    cm.save({**model, "content": external}, path)
    external = cm.get(path)["content"]
    await loader.maybe_notify()
    new_id = arbitrary_fid_manager.index(path)
    assert new_id != loader.file_id
    new_loader = FileLoader(new_id, arbitrary_fid_manager, cm)
    new_room = DocumentRoom(
        "new-room", model["format"], model["type"], new_loader, FakeEventLogger(), None, None
    )
    try:
        await new_room.initialize()
        loaded = await new_room._document.aget()
        assert (loaded if model["type"] == "file" else loaded["cells"][0]["source"]) == "external"
        assert await room._document.aget() == original
        assert room.outofband == {"originalPath": path}
        # Notifications from the preserved room must not detach the new disk session.
        await loader.maybe_notify()
        await room._on_outofband_change()
        assert arbitrary_fid_manager.get_id(path) == new_id
        with pytest.raises(OutOfBandChanges):
            await loader.maybe_save_content(model)
        assert cm.get(path)["content"] == external
    finally:
        await new_room.stop()
        await new_loader.clean()


def decode_status(message):
    decoder = Decoder(message)
    assert decoder.read_var_uint() == MessageType.RAW
    return json.loads(decoder.read_var_string())


async def test_external_change_status_broadcast_and_preserved(document):
    cm, loader, room, model, path = document
    clients = [MagicMock(send=AsyncMock()), MagicMock(send=AsyncMock())]
    room.clients.update(clients)
    cm.save({**model, "content": "external"}, path)
    await loader.maybe_notify()
    for client in clients:
        assert decode_status(client.send.call_args.args[0]) == {
            "type": "external-change",
            "change": {"originalPath": path},
        }
        client.send.assert_awaited_once()
    assert "outofband" not in room._document.ystate
    await loader.maybe_notify()
    for client in clients:
        client.send.assert_awaited_once()
    cm.save(model, path)
    await loader.maybe_notify()
    assert room.outofband == {"originalPath": path}
    for client in clients:
        client.send.assert_awaited_once()
    room.clients.clear()


@pytest.mark.parametrize("deleted", [False, True])
async def test_external_change_status_replayed_on_each_connection(document, jp_root_dir, deleted):
    cm, loader, room, model, path = document
    notice = {"originalPath": path}
    if deleted:
        (jp_root_dir / path).unlink()
        notice["reason"] = "deleted"
    else:
        cm.save({**model, "content": "external"}, path)
    await loader.maybe_notify()
    # The detached session must still replay status on reconnection.
    for _ in range(2):
        channel = MagicMock(send=AsyncMock())

        async def serve(client):
            assert client in room.clients
            client.send.assert_awaited_once()

        with patch.object(YRoom, "serve", side_effect=serve):
            await room.serve(channel)
        assert channel not in room.clients
        assert decode_status(channel.send.call_args.args[0]) == {
            "type": "external-change",
            "change": notice,
        }
    assert "outofband" not in room._document.ystate


@pytest.mark.parametrize("document", [False, True], indirect=True)
@pytest.mark.parametrize("trigger", ["poll", "save"])
@pytest.mark.parametrize("remove_mapping", [False, True])
async def test_external_deletion_preserves_content_and_notifies(
    document, jp_root_dir, arbitrary_fid_manager, trigger, remove_mapping
):
    cm, loader, room, model, path = document
    shared = await room._document.aget()
    client = MagicMock(send=AsyncMock())
    room.clients.add(client)
    (jp_root_dir / path).unlink()
    if remove_mapping:
        arbitrary_fid_manager.delete(path)
    try:
        if trigger == "poll":
            await loader.maybe_notify()
        else:
            await room._maybe_save_document(None, save_now=True)
        notice = {"originalPath": path, "reason": "deleted"}
        assert room.outofband == notice
        assert decode_status(client.send.call_args.args[0])["change"] == notice
        assert await room._document.aget() == shared
        assert room._document.dirty
        assert not cm.exists(path)
        await loader.maybe_notify()
        client.send.assert_awaited_once()
        with pytest.raises(OutOfBandChanges):
            await loader.maybe_save_content(model)
    finally:
        room.clients.clear()


async def test_deleted_file_restored_preserves_old_session(document, jp_root_dir):
    cm, loader, room, model, path = document
    shared = await room._document.aget()
    (jp_root_dir / path).unlink()
    await loader.maybe_notify()
    assert room.outofband["reason"] == "deleted"
    cm.save({**model, "content": "restored with new content"}, path)
    await loader.maybe_notify()
    assert room.outofband == {"originalPath": path}
    assert not loader.deleted
    assert await room._document.aget() == shared
    assert cm.get(path)["content"] == "restored with new content"
    with pytest.raises(OutOfBandChanges):
        await loader.maybe_save_content({**model, "content": shared})
    assert cm.get(path)["content"] == "restored with new content"


@pytest.mark.parametrize("document", [False, True], indirect=True)
async def test_explicit_overwrite_saves_snapshot_and_preserves_session(document):
    cm, loader, room, model, path = document
    shared = await room._document.aget()
    external = "disk content" if model["type"] == "file" else nbformat.v4.new_notebook()
    cm.save({**model, "content": external}, path)
    await loader.maybe_notify()
    assert room.outofband is not None
    cm.save({**model, "content": shared}, path)
    assert cm.get(path)["content"] == shared
    assert await room._document.aget() == shared
    assert room.outofband is not None
    with pytest.raises(OutOfBandChanges):
        await loader.maybe_save_content(model)


@pytest.mark.parametrize("behavior_on_external_change", ["overwrite"])
@pytest.mark.parametrize("document", [False, True], indirect=True)
@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("trigger", ["poll", "save"])
async def test_reload_mode_updates_document_without_changing_id(
    document, arbitrary_fid_manager, behavior_on_external_change, dirty, trigger
):
    cm, loader, room, model, path = document
    if dirty:
        if model["type"] == "file":
            room._document.source = "unsaved edits"
        else:
            edits = await room._document.aget()
            edits["cells"][0]["source"] = "unsaved edits"
            await room._document.aset(edits)
    external = (
        "external version"
        if model["type"] == "file"
        else nbformat.v4.new_notebook(
            cells=[nbformat.v4.new_code_cell("external")],
            metadata={
                "kernelspec": {"display_name": "Python 3", "name": "python3"},
                "language_info": {"name": "python"},
            },
        )
    )
    cm.save({**model, "content": external}, path)
    client = MagicMock(send=AsyncMock())
    room.clients.add(client)
    try:
        if trigger == "poll":
            await loader.maybe_notify()
        else:
            await room._maybe_save_document(None, save_now=True)
        assert await room._document.aget() == cm.get(path)["content"]
        assert room.outofband is None
        assert not room._document.dirty
        assert arbitrary_fid_manager.index(path) == loader.file_id
        assert room._document.hash == cm.get(path, content=False, require_hash=True)["hash"]
        client.send.assert_not_awaited()
        # Further edits can be saved using the same file loader and ID.
        saved = await loader.maybe_save_content({**model, "content": external})
        assert saved is not None
        assert arbitrary_fid_manager.index(path) == loader.file_id
    finally:
        room.clients.clear()


@pytest.mark.parametrize("behavior_on_external_change", ["overwrite"])
async def test_reload_mode_preserves_deleted_file_and_recovers(
    document, arbitrary_fid_manager, jp_root_dir, behavior_on_external_change
):
    cm, loader, room, model, path = document
    original = await room._document.aget()
    (jp_root_dir / path).unlink()
    await loader.maybe_notify()
    assert room.outofband == {"originalPath": path, "reason": "deleted"}
    assert await room._document.aget() == original
    assert arbitrary_fid_manager.index(path) == loader.file_id
    with pytest.raises(OutOfBandChanges):
        await loader.maybe_save_content(model)
    cm.save({**model, "content": "restored"}, path)
    await loader.maybe_notify()
    assert await room._document.aget() == "restored"
    assert room.outofband is None
    assert not room._document.dirty
    assert arbitrary_fid_manager.index(path) == loader.file_id


@pytest.mark.parametrize("behavior_on_external_change", ["overwrite"])
@pytest.mark.parametrize("failure", ["read", "apply"])
async def test_reload_mode_preserves_content_on_read_failure(
    document, arbitrary_fid_manager, behavior_on_external_change, failure
):
    cm, loader, room, model, path = document
    original = await room._document.aget()
    cm.save({**model, "content": "external"}, path)
    target, method = (loader, "load_content") if failure == "read" else (room._document, "aset")
    with patch.object(target, method, side_effect=OSError("Cannot load file")):
        for _ in range(2):
            await loader.maybe_notify()
            assert room.outofband == {"originalPath": path}
    assert await room._document.aget() == original
    assert room.outofband == {"originalPath": path}
    assert arbitrary_fid_manager.index(path) == loader.file_id
    with pytest.raises(OutOfBandChanges):
        await loader.maybe_save_content(model)
