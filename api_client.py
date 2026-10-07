from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


class APIClientError(Exception):
    """Raised when the FastAPI backend returns an error."""


@dataclass(frozen=True)
class Citation:
    source: str
    page_number: int | None
    chunk_index: int | None


@dataclass(frozen=True)
class QueryResult:
    answer: str
    citations: list[Citation]


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    text: str
    snippet: str
    source: str
    page_number: int | None
    chunk_index: int | None
    dense_score: float
    bm25_score: float
    hybrid_score: float
    rerank_score: float


@dataclass(frozen=True)
class SearchResponse:
    query: str
    knowledge_base_id: str
    results: list[SearchResult]


@dataclass(frozen=True)
class RecruiterEvidence:
    doc_id: str
    text: str
    snippet: str
    source: str
    page_number: int | None
    chunk_index: int | None
    dense_score: float
    bm25_score: float
    hybrid_score: float
    rerank_score: float


@dataclass(frozen=True)
class RecruiterCandidate:
    candidate_id: str
    full_name: str
    email: str | None
    location: str | None
    current_title: str | None
    current_company: str | None
    years_experience: float | None
    skills: list[str]
    availability: str | None
    resume_filename: str | None
    resume_document_id: str | None
    best_score: float
    evidence: list[RecruiterEvidence]


@dataclass(frozen=True)
class RecruiterSearchResponse:
    query: str
    knowledge_base_id: str
    results: list[RecruiterCandidate]


@dataclass(frozen=True)
class UploadResult:
    filename: str
    status: str
    doc_id: str | None
    page_count: int | None
    chunk_count: int | None
    error_message: str | None
    job_id: str | None = None


@dataclass(frozen=True)
class ReadinessResult:
    ready: bool
    status: str
    service: str
    checks: dict[str, bool]


class RAGAPIClient:
    def __init__(
        self,
        backend_url: str,
        timeout: int = 60,
    ) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.timeout = timeout

    def _headers(
        self,
        api_key: str | None = None,
    ) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
        }

        if api_key and api_key.strip():
            headers["X-API-Key"] = api_key.strip()

        return headers

    def _raise_for_error(
        self,
        response: requests.Response,
    ) -> None:
        if response.ok:
            return

        try:
            payload = response.json()
        except ValueError:
            payload = None

        detail: Any = payload

        if isinstance(payload, dict):
            detail = payload.get(
                "detail",
                payload,
            )

        raise APIClientError(
            f"Backend returned HTTP "
            f"{response.status_code}: {detail}"
        )

    def health(self) -> dict[str, Any]:
        try:
            response = requests.get(
                f"{self.backend_url}/health",
                headers=self._headers(),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise APIClientError(
                f"Unable to reach backend: {exc}"
            ) from exc

        self._raise_for_error(response)

        try:
            return response.json()
        except ValueError as exc:
            raise APIClientError(
                "Backend returned invalid JSON."
            ) from exc

    def readiness(
        self,
        api_key: str | None = None,
    ) -> ReadinessResult:
        try:
            response = requests.get(
                f"{self.backend_url}/health/ready",
                headers=self._headers(api_key),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise APIClientError(
                f"Unable to reach backend: {exc}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise APIClientError(
                "Backend returned invalid readiness JSON."
            ) from exc

        checks_payload = payload.get(
            "checks",
            {},
        )

        checks: dict[str, bool] = {}

        if isinstance(checks_payload, dict):
            for name, value in checks_payload.items():
                if isinstance(value, dict):
                    checks[name] = bool(
                        value.get("ok", False)
                    )
                else:
                    checks[name] = bool(value)

        return ReadinessResult(
            ready=bool(
                payload.get(
                    "status",
                    "not_ready",
                )
                == "ready"
            ),
            status=str(
                payload.get(
                    "status",
                    "unknown",
                )
            ),
            service=str(
                payload.get(
                    "service",
                    "unknown",
                )
            ),
            checks=checks,
        )

    # ---------------------------------------------------------
    # Level 1
    # ---------------------------------------------------------

    def query(
        self,
        question: str,
        knowledge_base_id: str,
        api_key: str,
    ) -> QueryResult:
        question = question.strip()
        knowledge_base_id = knowledge_base_id.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        if not knowledge_base_id:
            raise ValueError(
                "Knowledge base ID cannot be empty."
            )

        try:
            response = requests.post(
                f"{self.backend_url}/query",
                json={
                    "query": question,
                    "knowledge_base_id": knowledge_base_id,
                },
                headers=self._headers(api_key),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise APIClientError(
                f"Unable to reach backend: {exc}"
            ) from exc

        self._raise_for_error(response)

        try:
            payload = response.json()
        except ValueError as exc:
            raise APIClientError(
                "Backend returned invalid JSON."
            ) from exc

        citations: list[Citation] = []

        citations_payload = payload.get(
            "citations",
            [],
        )

        if isinstance(citations_payload, list):
            for item in citations_payload:
                if not isinstance(item, dict):
                    continue

                citations.append(
                    Citation(
                        source=str(
                            item.get(
                                "source",
                                "Unknown source",
                            )
                        ),
                        page_number=item.get(
                            "page_number"
                        ),
                        chunk_index=item.get(
                            "chunk_index"
                        ),
                    )
                )

        return QueryResult(
            answer=str(
                payload.get(
                    "answer",
                    "",
                )
            ),
            citations=citations,
        )

    # ---------------------------------------------------------
    # Level 2
    # ---------------------------------------------------------

    def search(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int,
        api_key: str,
    ) -> SearchResponse:
        query = query.strip()
        knowledge_base_id = knowledge_base_id.strip()

        if not query:
            raise ValueError(
                "Search query cannot be empty."
            )

        if not knowledge_base_id:
            raise ValueError(
                "Knowledge base ID cannot be empty."
            )

        try:
            response = requests.post(
                f"{self.backend_url}/search",
                json={
                    "query": query,
                    "knowledge_base_id": knowledge_base_id,
                    "top_k": top_k,
                },
                headers=self._headers(api_key),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise APIClientError(
                f"Unable to reach backend: {exc}"
            ) from exc

        self._raise_for_error(response)

        try:
            payload = response.json()
        except ValueError as exc:
            raise APIClientError(
                "Backend returned invalid JSON."
            ) from exc

        results: list[SearchResult] = []

        for item in payload.get("results", []):
            if not isinstance(item, dict):
                continue

            results.append(
                SearchResult(
                    doc_id=str(
                        item.get(
                            "doc_id",
                            "",
                        )
                    ),
                    text=str(
                        item.get(
                            "text",
                            "",
                        )
                    ),
                    snippet=str(
                        item.get(
                            "snippet",
                            item.get(
                                "text",
                                "",
                            ),
                        )
                    ),
                    source=str(
                        item.get(
                            "source",
                            "Unknown source",
                        )
                    ),
                    page_number=item.get(
                        "page_number"
                    ),
                    chunk_index=item.get(
                        "chunk_index"
                    ),
                    dense_score=float(
                        item.get(
                            "dense_score",
                            0.0,
                        )
                    ),
                    bm25_score=float(
                        item.get(
                            "bm25_score",
                            0.0,
                        )
                    ),
                    hybrid_score=float(
                        item.get(
                            "hybrid_score",
                            0.0,
                        )
                    ),
                    rerank_score=float(
                        item.get(
                            "rerank_score",
                            0.0,
                        )
                    ),
                )
            )

        return SearchResponse(
            query=str(
                payload.get(
                    "query",
                    query,
                )
            ),
            knowledge_base_id=str(
                payload.get(
                    "knowledge_base_id",
                    knowledge_base_id,
                )
            ),
            results=results,
        )

    # ---------------------------------------------------------
    # Level 3
    # ---------------------------------------------------------

    def recruiter_search(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int,
        api_key: str,
        skills: list[str] | None = None,
        min_years_experience: float | None = None,
        max_years_experience: float | None = None,
        location: str | None = None,
        title: str | None = None,
    ) -> RecruiterSearchResponse:
        query = query.strip()
        knowledge_base_id = knowledge_base_id.strip()

        if not query:
            raise ValueError(
                "Recruiter query cannot be empty."
            )

        if not knowledge_base_id:
            raise ValueError(
                "Knowledge base ID cannot be empty."
            )

        payload_request: dict[str, Any] = {
            "query": query,
            "knowledge_base_id": knowledge_base_id,
            "top_k": top_k,
            "skills": skills or [],
        }

        if min_years_experience is not None:
            payload_request[
                "min_years_experience"
            ] = min_years_experience

        if max_years_experience is not None:
            payload_request[
                "max_years_experience"
            ] = max_years_experience

        if location and location.strip():
            payload_request["location"] = (
                location.strip()
            )

        if title and title.strip():
            payload_request["title"] = (
                title.strip()
            )

        try:
            response = requests.post(
                f"{self.backend_url}/recruiter/search",
                json=payload_request,
                headers=self._headers(api_key),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise APIClientError(
                f"Unable to reach backend: {exc}"
            ) from exc

        self._raise_for_error(response)

        try:
            payload = response.json()
        except ValueError as exc:
            raise APIClientError(
                "Backend returned invalid JSON."
            ) from exc

        candidates: list[RecruiterCandidate] = []

        for item in payload.get("results", []):
            if not isinstance(item, dict):
                continue

            evidence_items: list[
                RecruiterEvidence
            ] = []

            for evidence in item.get(
                "evidence",
                [],
            ):
                if not isinstance(evidence, dict):
                    continue

                evidence_items.append(
                    RecruiterEvidence(
                        doc_id=str(
                            evidence.get(
                                "doc_id",
                                "",
                            )
                        ),
                        text=str(
                            evidence.get(
                                "text",
                                "",
                            )
                        ),
                        snippet=str(
                            evidence.get(
                                "snippet",
                                evidence.get(
                                    "text",
                                    "",
                                ),
                            )
                        ),
                        source=str(
                            evidence.get(
                                "source",
                                "Unknown source",
                            )
                        ),
                        page_number=evidence.get(
                            "page_number"
                        ),
                        chunk_index=evidence.get(
                            "chunk_index"
                        ),
                        dense_score=float(
                            evidence.get(
                                "dense_score",
                                0.0,
                            )
                        ),
                        bm25_score=float(
                            evidence.get(
                                "bm25_score",
                                0.0,
                            )
                        ),
                        hybrid_score=float(
                            evidence.get(
                                "hybrid_score",
                                0.0,
                            )
                        ),
                        rerank_score=float(
                            evidence.get(
                                "rerank_score",
                                0.0,
                            )
                        ),
                    )
                )

            candidates.append(
                RecruiterCandidate(
                    candidate_id=str(
                        item.get(
                            "candidate_id",
                            "",
                        )
                    ),
                    full_name=str(
                        item.get(
                            "full_name",
                            "Unknown candidate",
                        )
                    ),
                    email=item.get("email"),
                    location=item.get(
                        "location"
                    ),
                    current_title=item.get(
                        "current_title"
                    ),
                    current_company=item.get(
                        "current_company"
                    ),
                    years_experience=item.get(
                        "years_experience"
                    ),
                    skills=list(
                        item.get(
                            "skills",
                            [],
                        )
                        or []
                    ),
                    availability=item.get(
                        "availability"
                    ),
                    resume_filename=item.get(
                        "resume_filename"
                    ),
                    resume_document_id=item.get(
                        "resume_document_id"
                    ),
                    best_score=float(
                        item.get(
                            "best_score",
                            0.0,
                        )
                    ),
                    evidence=evidence_items,
                )
            )

        return RecruiterSearchResponse(
            query=str(
                payload.get(
                    "query",
                    query,
                )
            ),
            knowledge_base_id=str(
                payload.get(
                    "knowledge_base_id",
                    knowledge_base_id,
                )
            ),
            results=candidates,
        )

    # ---------------------------------------------------------
    # Upload / ingestion
    # ---------------------------------------------------------

    def upload_document(
        self,
        file_name: str,
        file_bytes: bytes,
        api_key: str,
        knowledge_base_id: str,
        content_type: str | None = None,
    ) -> UploadResult:
        if not file_name.strip():
            raise ValueError(
                "File name cannot be empty."
            )

        if not file_bytes:
            raise ValueError(
                "File cannot be empty."
            )

        if not knowledge_base_id.strip():
            raise ValueError(
                "Knowledge base ID cannot be empty."
            )

        files = {
            "file": (
                file_name,
                file_bytes,
                content_type
                or "application/octet-stream",
            )
        }

        data = {
            "knowledge_base_id":
                knowledge_base_id.strip(),
        }

        try:
            response = requests.post(
                f"{self.backend_url}/documents/upload",
                files=files,
                data=data,
                headers=self._headers(api_key),
                timeout=max(
                    self.timeout,
                    300,
                ),
            )
        except requests.RequestException as exc:
            raise APIClientError(
                f"Unable to reach backend: {exc}"
            ) from exc

        self._raise_for_error(response)

        try:
            payload = response.json()
        except ValueError as exc:
            raise APIClientError(
                "Backend returned invalid JSON."
            ) from exc

        return UploadResult(
            job_id=str(
                payload.get(
                    "job_id",
                    "",
                )
            ),
            filename=str(
                payload.get(
                    "filename",
                    file_name,
                )
            ),
            status=str(
                payload.get(
                    "status",
                    "unknown",
                )
            ),
            doc_id=payload.get("doc_id"),
            page_count=payload.get(
                "page_count"
            ),
            chunk_count=payload.get(
                "chunk_count"
            ),
            
            error_message=payload.get(
                "error_message"
            ),
        )

    def get_ingestion_job(
        self,
        job_id: str,
        api_key: str,
    ) -> UploadResult:
        try:
            response = requests.get(
                f"{self.backend_url}/documents/jobs/{job_id}",
                headers=self._headers(api_key),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise APIClientError(
                f"Unable to reach backend: {exc}"
            ) from exc

        self._raise_for_error(response)

        try:
            payload = response.json()
        except ValueError as exc:
            raise APIClientError(
                "Backend returned invalid JSON."
            ) from exc

        return UploadResult(
            job_id=str(
                payload.get(
                    "job_id",
                    job_id,
                )
            ),
            filename=str(
                payload.get(
                    "filename",
                    "",
                )
            ),
            status=str(
                payload.get(
                    "status",
                    "unknown",
                )
            ),
            doc_id=payload.get(
                "doc_id"
            ),
            page_count=payload.get(
                "page_count"
            ),
            chunk_count=payload.get(
                "chunk_count"
            ),
            error_message=payload.get(
                "error_message"
            ),
        )

