"""Utility package for RealTimeService.

This package contains small helper modules used across the realtime pipeline
and alerting code. Files here are intentionally lightweight wrappers that do
not change existing functionality.
"""

from .config import get_env, EMAIL_FROM, EMAIL_TO, SMTP_SERVER, SMTP_PORT
from .file_utils import read_json, write_json
from .logging_setup import setup_logging

__all__ = [
    "get_env",
    "EMAIL_FROM",
    "EMAIL_TO",
    "SMTP_SERVER",
    "SMTP_PORT",
    "read_json",
    "write_json",
    "setup_logging",
]
