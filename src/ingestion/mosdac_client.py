"""
Official MOSDAC API Client for INSAT-3DS / INSAT-3DR Satellite Feeds.
Implements the exact ISRO MOSDAC protocol (gettoken, datasets.json, download API)
matching the official MOSDAC data download tool (mdapi.py & config.json).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import requests


logger = logging.getLogger("blink.mosdac_client")


@dataclass
class MOSDACObservation:
    """Represents an available MOSDAC satellite observation granule."""
    satellite_id: str
    dataset_id: str
    record_id: str
    identifier: str
    prod_date: str
    timestamp_utc: str
    timestamp_ist: str
    local_path: Optional[str]
    is_cached: bool


class MOSDACClient:
    """
    Client for ISRO Meteorological & Oceanographic Satellite Data Archival Centre (MOSDAC).
    Implements authentication (gettoken), catalogue search (datasets.json), and binary
    HDF5 streaming (download API) according to official ISRO specifications.
    """

    TOKEN_URL = "https://mosdac.gov.in/download_api/gettoken"
    SEARCH_URL = "https://mosdac.gov.in/apios/datasets.json"
    CHECK_INTERNET_URL = "https://mosdac.gov.in/download_api/check-internet"
    DOWNLOAD_URL = "https://mosdac.gov.in/download_api/download"
    REFRESH_URL = "https://mosdac.gov.in/download_api/refresh-token"
    LOGOUT_URL = "https://mosdac.gov.in/download_api/logout"

    DEFAULT_DATASET_ID = "3SIMG_L1B_STD"      # INSAT-3DS Level-1B Standard Imager
    DEFAULT_INSAT3DR_DATASET_ID = "3RIMG_L1B_STD"  # INSAT-3DR Level-1B Standard Imager

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = "config.json",
        username: Optional[str] = None,
        password: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.config_path = Path(config_path) if config_path else Path("config.json")
        self.cache_dir = Path(cache_dir or "data/raw_netcdf/mosdac")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.username = username or ""
        self.password = password or ""
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None

        # Load from config.json if present
        self._load_from_config_file()

    def _load_from_config_file(self) -> bool:
        """Loads credentials from config.json if available."""
        if not self.config_path.exists():
            return False
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw = f.read()
            # Windows path preprocessing
            fixed = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r'\\\\', raw)
            data = json.loads(fixed)

            creds = data.get("user_credentials", {})
            user = creds.get("username/email") or creds.get("username")
            pwd = creds.get("password")

            if user and user != "your_username":
                self.username = user
            if pwd and pwd != "your_password":
                self.password = pwd

            dl_settings = data.get("download_settings", {})
            custom_dl_path = dl_settings.get("download_path")
            if custom_dl_path and custom_dl_path != "/home/sys_oper/MOSDAC_Downloads/":
                self.cache_dir = Path(custom_dl_path)
                self.cache_dir.mkdir(parents=True, exist_ok=True)

            return True
        except Exception as e:
            logger.warning(f"Could not load config.json: {e}")
            return False

    @property
    def is_configured(self) -> bool:
        """Returns True if user has configured valid non-default credentials."""
        return bool(
            self.username
            and self.password
            and self.username != "your_username"
            and self.password != "your_password"
        )

    def set_credentials(self, username: str, password: str) -> None:
        """Sets and persists credentials to config.json."""
        self.username = username.strip()
        self.password = password.strip()
        self.access_token = None
        self.refresh_token = None

        # Update or create config.json
        config_data = {
            "user_credentials": {
                "username/email": self.username,
                "password": self.password,
            },
            "search_parameters": {
                "datasetId": self.DEFAULT_DATASET_ID,
                "startTime": datetime.utcnow().strftime("%Y-%m-%d"),
                "endTime": datetime.utcnow().strftime("%Y-%m-%d"),
                "count": "",
                "boundingBox": "",
                "gId": "",
            },
            "download_settings": {
                "download_path": str(self.cache_dir.resolve()).replace("\\", "/"),
                "organize_by_date": False,
                "skip_user_input": True,
                "generate_error_logs": True,
                "error_logs_dir": "error_logs",
            },
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)

    def authenticate(self) -> bool:
        """
        Authenticates with MOSDAC token endpoint:
        POST https://mosdac.gov.in/download_api/gettoken
        """
        if not self.is_configured:
            return False

        payload = {"username": self.username, "password": self.password}
        try:
            resp = requests.post(self.TOKEN_URL, json=payload, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                self.access_token = data.get("access_token")
                self.refresh_token = data.get("refresh_token")
                return bool(self.access_token)
            else:
                logger.warning(f"MOSDAC Auth failed with status {resp.status_code}: {resp.text}")
                return False
        except Exception as e:
            logger.warning(f"MOSDAC Auth network exception: {e}")
            return False

    def search_live_catalog(
        self,
        dataset_id: str = DEFAULT_DATASET_ID,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        count: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Queries official MOSDAC search endpoint:
        GET https://mosdac.gov.in/apios/datasets.json
        """
        params = {"datasetId": dataset_id}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        if count:
            params["count"] = str(count)

        try:
            resp = requests.get(self.SEARCH_URL, params=params, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                entries = data.get("entries", [])
                results = []
                for item in entries:
                    results.append({
                        "id": item.get("id"),
                        "identifier": item.get("identifier"),
                        "updated": item.get("updated"),
                        "datasetId": dataset_id,
                    })
                return results
        except Exception as e:
            logger.warning(f"MOSDAC search catalog failed: {e}")
        return []

    def download_file(self, record_id: str, identifier: str) -> Optional[Path]:
        """
        Downloads a specific HDF5 granule using Bearer token:
        GET https://mosdac.gov.in/download_api/download?id=<record_id>
        """
        if not self.access_token:
            if not self.authenticate():
                return None

        dest_file = self.cache_dir / identifier
        if dest_file.exists():
            return dest_file

        headers = {"Authorization": f"Bearer {self.access_token}"}
        params = {"id": record_id}

        try:
            resp = requests.get(self.DOWNLOAD_URL, headers=headers, params=params, stream=True, timeout=15)
            if resp.status_code == 401 and self.refresh_token:
                # Refresh token and retry
                ref_resp = requests.post(self.REFRESH_URL, json={"refresh_token": self.refresh_token}, timeout=8)
                if ref_resp.status_code == 200:
                    self.access_token = ref_resp.json().get("access_token")
                    headers = {"Authorization": f"Bearer {self.access_token}"}
                    resp = requests.get(self.DOWNLOAD_URL, headers=headers, params=params, stream=True, timeout=15)

            if resp.status_code == 200:
                tmp_file = self.cache_dir / (identifier + ".part")
                with open(tmp_file, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1048576):
                        if chunk:
                            f.write(chunk)
                os.rename(tmp_file, dest_file)
                return dest_file
            else:
                logger.warning(f"MOSDAC download failed with status {resp.status_code}")
        except Exception as e:
            logger.warning(f"MOSDAC download exception: {e}")
        return None

    def query_available_scans(
        self,
        date_str: str,
        start_hour_utc: int = 0,
        end_hour_utc: int = 23,
        satellite_id: str = "INSAT-3DS",
    ) -> List[Dict[str, Any]]:
        """
        Queries or generates synoptic 15-minute scan list for a given date.
        If live catalog is reachable, complements with real MOSDAC granule IDs.
        """
        scans = []
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            target_date = datetime.utcnow().date()

        dataset_id = self.DEFAULT_DATASET_ID if satellite_id == "INSAT-3DS" else self.DEFAULT_INSAT3DR_DATASET_ID

        # Try live catalog search
        live_entries = self.search_live_catalog(
            dataset_id=dataset_id,
            start_time=date_str,
            end_time=date_str,
        )
        live_map = {e["identifier"]: e["id"] for e in live_entries}

        for hour in range(start_hour_utc, min(24, end_hour_utc + 1)):
            for minute in [0, 15, 30, 45]:
                dt_utc = datetime(target_date.year, target_date.month, target_date.day, hour, minute)
                dt_ist = dt_utc + timedelta(hours=5, minutes=30)

                prefix = "3SIMG" if satellite_id == "INSAT-3DS" else "3RIMG"
                file_name = f"{prefix}_{dt_utc.strftime('%d%b%Y_%H%M')}_L1B_STD_V01R00.h5".upper()

                local_path = self.cache_dir / file_name
                is_cached = local_path.exists()
                record_id = live_map.get(file_name)

                scans.append({
                    "satellite_id": satellite_id,
                    "dataset_id": dataset_id,
                    "record_id": record_id,
                    "timestamp_utc": dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "timestamp_ist": dt_ist.strftime("%d-%m-%Y %H:%M IST"),
                    "file_name": file_name,
                    "local_path": str(local_path) if is_cached else None,
                    "is_cached": is_cached,
                    "channels": ["IMG_VIS", "IMG_SWIR", "IMG_MWIR", "IMG_WV", "IMG_TIR1", "IMG_TIR2"],
                    "spatial_resolution_km": {"VIS": 1.0, "SWIR": 1.0, "MIR": 4.0, "WV": 8.0, "TIR": 4.0},
                    "coverage": {"lat_north": 35.0, "lat_south": 5.0, "lon_west": 65.0, "lon_east": 100.0},
                })

        return scans

    def fetch_scan_pair(
        self,
        date_str: str,
        t0_time_str: str,
        cadence_minutes: int = 15,
        satellite_id: str = "INSAT-3DS",
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Pairs two consecutive observations T0 and T1, downloading real HDF5 if authenticated.
        """
        try:
            hour, minute = map(int, t0_time_str.split(":"))
            t0_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=hour, minute=minute)
        except Exception:
            t0_dt = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

        t1_dt = t0_dt + timedelta(minutes=cadence_minutes)

        scans_t0 = self.query_available_scans(
            t0_dt.strftime("%Y-%m-%d"),
            start_hour_utc=t0_dt.hour,
            end_hour_utc=t0_dt.hour,
            satellite_id=satellite_id,
        )
        scans_t1 = self.query_available_scans(
            t1_dt.strftime("%Y-%m-%d"),
            start_hour_utc=t1_dt.hour,
            end_hour_utc=t1_dt.hour,
            satellite_id=satellite_id,
        )

        match_0 = next(
            (s for s in scans_t0 if s["timestamp_utc"].endswith(t0_dt.strftime("%H:%M:00Z"))),
            scans_t0[0] if scans_t0 else None,
        )
        match_1 = next(
            (s for s in scans_t1 if s["timestamp_utc"].endswith(t1_dt.strftime("%H:%M:00Z"))),
            scans_t1[0] if scans_t1 else None,
        )

        # If authenticated and record_ids are known, download HDF5 files
        if self.is_configured and match_0 and match_0.get("record_id"):
            dl0 = self.download_file(match_0["record_id"], match_0["file_name"])
            if dl0:
                match_0["local_path"] = str(dl0)
                match_0["is_cached"] = True

        if self.is_configured and match_1 and match_1.get("record_id"):
            dl1 = self.download_file(match_1["record_id"], match_1["file_name"])
            if dl1:
                match_1["local_path"] = str(dl1)
                match_1["is_cached"] = True

        return match_0, match_1
