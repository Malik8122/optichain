"""Parallel, cached rolling-origin backtest.

One task = (item, model, origin). Each task is cached on disk under a key that hashes
everything that affects its result (series up to the origin, model, identification
record, grid, window, horizon, levels), so re-runs only refit what changed.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .data_pipeline import d_to_date
from .evaluation import rolling_origin

CACHE_VERSION = "m7-v1"  # bump to invalidate every cached fit after a model-code change


@dataclass
class Task:
    item: str
    model: str
    origin: int
    y: pd.Series  # full series; rolling_origin slices it at the origin
    h: int
    fit_window_days: int
    levels: tuple
    context: dict = field(default_factory=dict)

    def key(self) -> str:
        y_train = self.y.loc[: d_to_date(self.origin)].iloc[-self.fit_window_days:]
        blob = json.dumps({
            "v": CACHE_VERSION, "item": self.item, "model": self.model, "origin": self.origin, "h": self.h,
            "win": self.fit_window_days, "levels": list(self.levels), "ctx": self.context,
            "y": hashlib.sha1(y_train.to_numpy().tobytes()).hexdigest(),
            "y_end": str(y_train.index[-1]),
        }, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _run(task: Task) -> tuple[pd.DataFrame, dict, float]:
    t0 = time.time()
    fc, params = rolling_origin(task.y, task.model, [task.origin], task.h, task.fit_window_days,
                                task.levels, context_by_origin={task.origin: task.context})
    return fc.assign(item_id=task.item), params[0], time.time() - t0


def run_tasks(tasks: list[Task], cache_dir: Path, workers: int | None = None, log=print):
    """Run tasks (cached, in parallel). Returns (forecasts, params dict, timing frame)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    done, todo = {}, []
    for t in tasks:
        path = cache_dir / f"{t.item}_{t.model}_{t.origin}_{t.key()}.pkl"
        if path.exists():
            done[id(t)] = pickle.loads(path.read_bytes()) + (True,)
        else:
            todo.append((t, path))
    log(f"{len(tasks)} tasks: {len(tasks) - len(todo)} cached, {len(todo)} to fit")

    if todo:
        workers = workers or max(1, (os.cpu_count() or 2) - 1)
        # heaviest first so long SARIMA searches start early
        weight = {"sarima": 3, "arima": 2, "ets": 1}
        todo.sort(key=lambda tp: -weight.get(tp[0].model, 0))
        t_start = time.time()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run, t): (t, path) for t, path in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                t, path = futs[fut]
                fc, params, secs = fut.result()
                path.write_bytes(pickle.dumps((fc, params, secs)))
                done[id(t)] = (fc, params, secs, False)
                if i % 10 == 0 or i == len(todo):
                    log(f"  {i}/{len(todo)} fitted ({time.time() - t_start:.0f}s elapsed)")

    frames, params, timing = [], {}, []
    for t in tasks:
        fc, p, secs, cached = done[id(t)]
        frames.append(fc)
        params.setdefault(f"{t.item}/{t.model}", []).append(p)
        timing.append({"item_id": t.item, "model": t.model, "origin_d": t.origin, "seconds": secs, "cached": cached})
    return pd.concat(frames, ignore_index=True), params, pd.DataFrame(timing)
