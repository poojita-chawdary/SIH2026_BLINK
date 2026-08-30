"""
Tests for MOSDACClient.
"""

from datetime import datetime
from pathlib import Path
import pytest

from src.ingestion.mosdac_client import MOSDACClient


def test_mosdac_client_config(tmp_path):
    config_file = tmp_path / "config.json"
    client = MOSDACClient(config_path=config_file, cache_dir=tmp_path)
    assert not client.is_configured

    client.set_credentials("test_user@domain.in", "sec_password_12345")
    assert client.is_configured
    assert client.username == "test_user@domain.in"
    assert client.password == "sec_password_12345"
    assert config_file.exists()


def test_mosdac_query_available_scans(tmp_path):
    config_file = tmp_path / "config.json"
    client = MOSDACClient(config_path=config_file, cache_dir=tmp_path)
    scans = client.query_available_scans("2026-08-28", start_hour_utc=10, end_hour_utc=11)

    # 2 hours * 4 scans/hour = 8 scans
    assert len(scans) == 8
    assert "IMG_TIR1" in scans[0]["channels"]
    assert "3SIMG" in scans[0]["file_name"]
    assert scans[0]["satellite_id"] == "INSAT-3DS"


def test_mosdac_fetch_scan_pair(tmp_path):
    config_file = tmp_path / "config.json"
    client = MOSDACClient(config_path=config_file, cache_dir=tmp_path)
    s0, s1 = client.fetch_scan_pair("2026-08-28", "10:00", cadence_minutes=15)

    assert s0 is not None
    assert s1 is not None
    assert "10:00:00Z" in s0["timestamp_utc"]
    assert "10:15:00Z" in s1["timestamp_utc"]
