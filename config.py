from __future__ import annotations

import os
from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class FrontendConfig:
    backend_url: str
    request_timeout: int = 60

    # Three application knowledge bases
    level1_knowledge_base_id: str = "kb_default"
    level2_knowledge_base_id: str = "kb_company"
    level3_knowledge_base_id: str = "kb_recruitment"


def _get_secret_or_env(
    name: str,
    default: str = "",
) -> str:
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None

    if value:
        return str(value)

    return os.getenv(name, default)


def get_config() -> FrontendConfig:
    backend_url = _get_secret_or_env(
        "BACKEND_URL",
        "http://127.0.0.1:8000",
    ).strip().rstrip("/")

    if not backend_url:
        raise ValueError(
            "BACKEND_URL must not be empty."
        )

    return FrontendConfig(
        backend_url=backend_url,
        request_timeout=int(
            _get_secret_or_env(
                "BACKEND_TIMEOUT",
                "60",
            )
        ),
        level1_knowledge_base_id=_get_secret_or_env(
            "LEVEL1_KNOWLEDGE_BASE_ID",
            "kb_default",
        ).strip(),
        level2_knowledge_base_id=_get_secret_or_env(
            "LEVEL2_KNOWLEDGE_BASE_ID",
            "kb_company",
        ).strip(),
        level3_knowledge_base_id=_get_secret_or_env(
            "LEVEL3_KNOWLEDGE_BASE_ID",
            "kb_recruitment",
        ).strip(),
    )

