"""Configuration management and structured logging for the Quant Research Platform."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StructuredJsonFormatter(logging.Formatter):
    """Structured JSON formatter for reproducible research logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", json_logs: bool = False) -> logging.Logger:
    """Configure application-wide logger."""
    logger = logging.getLogger("quant_platform")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        if json_logs:
            handler.setFormatter(StructuredJsonFormatter())
        else:
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
            )
        logger.addHandler(handler)
    return logger


class PlatformSettings(BaseSettings):
    """Typed environment settings for the quantitative research platform."""

    model_config = SettingsConfigDict(
        env_prefix="QUANT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("./data_store"))
    duckdb_path: Path = Field(default=Path("./data_store/quant_research.duckdb"))
    log_level: str = Field(default="INFO")
    default_base_currency: Literal["AUD", "USD"] = Field(default="AUD")

    # Tax and execution parameters
    au_corporate_tax_rate: float = Field(default=0.30, ge=0.0, le=0.5)
    us_dividend_withholding_tax: float = Field(default=0.15, ge=0.0, le=0.5)
    default_commission_bps: float = Field(default=8.0, ge=0.0)
    default_slippage_bps: float = Field(default=5.0, ge=0.0)

    # AI Agent configuration
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", validation_alias="GEMINI_MODEL")

    def ensure_dirs(self) -> None:
        """Create required storage directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "parquet").mkdir(parents=True, exist_ok=True)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)


def get_settings() -> PlatformSettings:
    """Load and initialize platform settings."""
    settings = PlatformSettings()
    settings.ensure_dirs()
    return settings
