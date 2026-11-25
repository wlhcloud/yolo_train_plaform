"""Atomic, local-only inference snapshots paired with an audit byte offset."""

import hashlib
import json
import os
import pickle
import threading
from pathlib import Path

_id_lock = threading.Lock()


def reserve_track_ids(minimum=0):
    from boxmot.trackers.bytetrack.basetrack import BaseTrack

    with _id_lock:
        if not getattr(BaseTrack, "_road_ids_locked", False):

            def next_id():
                with _id_lock:
                    BaseTrack._count += 1
                    return BaseTrack._count

            BaseTrack.next_id = staticmethod(next_id)
            BaseTrack._road_ids_locked = True
        BaseTrack._count = max(BaseTrack._count, minimum)


def signature(config, stream, source, model):
    weight = getattr(model, "ckpt_path", None)
    identity = None
    if weight and Path(weight).is_file():
        stat = Path(weight).stat()
        identity = [str(Path(weight).resolve()), stat.st_size, stat.st_mtime_ns]
    value = [1, config, stream, source, model.names, identity]
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class StreamCheckpoint:
    def __init__(self, path, fingerprint):
        self.path = Path(path)
        self.fingerprint = fingerprint

    def load(self):
        if not self.path.exists():
            return None
        # Only read the application's own snapshot, never an uploaded checkpoint.
        try:
            with self.path.open("rb") as handle:
                state = pickle.load(handle)
        except Exception as error:
            raise ValueError(
                "帧断点读取失败，请使用重新运行，避免遗漏或重复计数"
            ) from error
        if state.get("signature") != self.fingerprint:
            return None
        return state

    def save(self, state, audit):
        audit.flush()
        os.fsync(audit.fileno())
        state = {**state, "signature": self.fingerprint, "audit_offset": audit.tell()}
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
