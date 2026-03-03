from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests


TUSHARE_API_URL = "http://api.tushare.pro"


@dataclass
class TushareClient:
    token: str
    api_url: str = TUSHARE_API_URL
    timeout: int = 30

    @classmethod
    def from_env(cls) -> "TushareClient":
        token = os.environ.get("TUSHARE_TOKEN") or load_token_from_dotenv()
        if not token:
            raise ValueError("TUSHARE_TOKEN is not set and .env does not contain it.")
        return cls(token=token)

    def query(self, api_name: str, params: dict[str, Any], fields: list[str]) -> pd.DataFrame:
        payload = {
            "api_name": api_name,
            "token": self.token,
            "params": params,
            "fields": ",".join(fields),
        }
        response = requests.post(self.api_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        code = result.get("code", -1)
        if code != 0:
            msg = result.get("msg", "Unknown Tushare error")
            raise RuntimeError(f"Tushare API error {code}: {msg}")

        data = result.get("data", {})
        return pd.DataFrame(data.get("items", []), columns=data.get("fields", []))

    def daily_stock(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        fields = [
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "vol",
            "amount",
        ]
        return self.query(
            api_name="daily",
            params={
                "ts_code": ts_code,
                "start_date": start_date,
                "end_date": end_date,
            },
            fields=fields,
        )

    def daily_index(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        fields = [
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "vol",
            "amount",
        ]
        return self.query(
            api_name="index_daily",
            params={
                "ts_code": ts_code,
                "start_date": start_date,
                "end_date": end_date,
            },
            fields=fields,
        )


def load_token_from_dotenv() -> str | None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "TUSHARE_TOKEN":
            return value.strip().strip("'\"")
    return None
