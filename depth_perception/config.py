"""Strict YAML configuration loaders for the depth baseline."""

from __future__ import annotations

from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .types import CameraIntrinsics, DepthDetectorConfig


T = TypeVar("T")


def _load_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    content = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"{source} must contain a YAML mapping")
    return content


def _construct_strict(model: type[T], values: dict[str, Any], source: str | Path) -> T:
    allowed = {item.name for item in fields(model)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown keys in {source}: {', '.join(unknown)}")
    missing = [
        item.name
        for item in fields(model)
        if item.default is MISSING and item.default_factory is MISSING and item.name not in values
    ]
    if missing:
        raise ValueError(f"missing keys in {source}: {', '.join(missing)}")
    try:
        return model(**values)
    except TypeError as error:
        raise ValueError(f"invalid keys in {source}: {error}") from error


def load_config(path: str | Path) -> DepthDetectorConfig:
    return _construct_strict(DepthDetectorConfig, _load_mapping(path), path)


def load_intrinsics(path: str | Path) -> CameraIntrinsics:
    return _construct_strict(CameraIntrinsics, _load_mapping(path), path)
