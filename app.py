from __future__ import annotations

import time

import streamlit as st

from api_client import (
    APIClientError,
    Citation,
    RAGAPIClient,
    RecruiterCandidate,
    RecruiterEvidence,
    SearchResult,
)
from config import get_config


st.set_page_config(
    page_title="Production RAG Platform",
    page_icon="🔎",
    layout="wide",
)


@st.cache_resource
def get_api_client() -> RAGAPIClient:
    config = get_config()

    return RAGAPIClient(
        backend_url=config.backend_url,
        timeout=config.request_timeout,
    )


def initialize_session_state() -> None:
    defaults = {
        "api_key": "",
        "backend_status": None,
        "level1_messages": [],
        "level2_results": None,
        "level3_results": None,
        "level2_query": "",
        "level3_query": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# Common rendering helpers
# ============================================================

def render_citations(
    citations: list[Citation],
) -> None:
    if not citations:
        st.caption("No citations returned.")
        return

    st.markdown("### 📚 Sources")

    for index, citation in enumerate(
        citations,
        start=1,
    ):
        page_text = (
            f"Page {citation.page_number}"
            if citation.page_number is not None
            else "Page unavailable"
        )

        chunk_text = (
            f"Chunk {citation.chunk_index}"
            if citation.chunk_index is not None
            else "Chunk unavailable"
        )

        st.markdown(
            f"**{index}. {citation.source}**  \n"
            f"{page_text} · {chunk_text}"
        )


def render_score_metrics(
    result: SearchResult,
) -> None:
    columns = st.columns(4)

    columns[0].metric(
        "Dense",
        f"{result.dense_score:.3f}",
    )

    columns[1].metric(
        "BM25",
        f"{result.bm25_score:.3f}",
    )

    columns[2].metric(
        "Hybrid",
        f"{result.hybrid_score:.3f}",
    )

    columns[3].metric(
        "Rerank",
        f"{result.rerank_score:.3f}",
    )


def render_level2_result(
    index: int,
    result: SearchResult,
) -> None:
    with st.container(border=True):
        st.markdown(
            f"### {index}. {result.source}"
        )

        location_parts = []

        if result.page_number is not None:
            location_parts.append(
                f"Page {result.page_number}"
            )

        if result.chunk_index is not None:
            location_parts.append(
                f"Chunk {result.chunk_index}"
            )

        if location_parts:
            st.caption(
                " · ".join(location_parts)
            )

        st.markdown(result.snippet)

        render_score_metrics(result)

        if result.doc_id:
            st.caption(
                f"Document ID: `{result.doc_id}`"
            )


def render_recruiter_evidence(
    evidence: RecruiterEvidence,
) -> None:
    with st.expander(
        f"📄 {evidence.source}"
    ):
        location_parts = []

        if evidence.page_number is not None:
            location_parts.append(
                f"Page {evidence.page_number}"
            )

        if evidence.chunk_index is not None:
            location_parts.append(
                f"Chunk {evidence.chunk_index}"
            )

        if location_parts:
            st.caption(
                " · ".join(location_parts)
            )

        st.markdown(evidence.snippet)

        columns = st.columns(4)

        columns[0].metric(
            "Dense",
            f"{evidence.dense_score:.3f}",
        )

        columns[1].metric(
            "BM25",
            f"{evidence.bm25_score:.3f}",
        )

        columns[2].metric(
            "Hybrid",
            f"{evidence.hybrid_score:.3f}",
        )

        columns[3].metric(
            "Rerank",
            f"{evidence.rerank_score:.3f}",
        )

        if evidence.doc_id:
            st.caption(
                f"Document ID: `{evidence.doc_id}`"
            )


def render_recruiter_candidate(
    index: int,
    candidate: RecruiterCandidate,
) -> None:
    with st.container(border=True):
        st.markdown(
            f"## {index}. {candidate.full_name}"
        )

        if candidate.current_title:
            st.markdown(
                f"**{candidate.current_title}**"
            )

        profile_columns = st.columns(4)

        profile_columns[0].metric(
            "Experience",
            (
                f"{candidate.years_experience:g} years"
                if candidate.years_experience
                is not None
                else "—"
            ),
        )

        profile_columns[1].metric(
            "Match score",
            f"{candidate.best_score:.3f}",
        )

        profile_columns[2].metric(
            "Location",
            candidate.location or "—",
        )

        profile_columns[3].metric(
            "Availability",
            candidate.availability or "—",
        )

        if candidate.current_company:
            st.write(
                f"**Company:** "
                f"{candidate.current_company}"
            )

        if candidate.skills:
            st.write(
                "**Skills:** "
                + " · ".join(candidate.skills)
            )

        if candidate.email:
            st.write(
                f"**Email:** {candidate.email}"
            )

        if candidate.resume_filename:
            st.caption(
                f"Resume: {candidate.resume_filename}"
            )

        if candidate.resume_document_id:
            st.caption(
                "Resume document ID: "
                f"`{candidate.resume_document_id}`"
            )

        st.markdown("### 🔎 Resume evidence")

        if not candidate.evidence:
            st.info(
                "No resume evidence was returned."
            )
        else:
            for evidence in candidate.evidence:
                render_recruiter_evidence(
                    evidence
                )


# ============================================================
# Sidebar
# ============================================================

def render_sidebar(
    client: RAGAPIClient,
) -> None:
    config = get_config()

    with st.sidebar:
        st.header("⚙️ RAG Platform")

        st.subheader("🔐 Authentication")

        api_key = st.text_input(
            "API Key",
            value=st.session_state.api_key,
            type="password",
            help=(
                "Enter your FastAPI API key. "
                "User keys can search. "
                "Admin keys can upload documents."
            ),
        )

        st.session_state.api_key = api_key

        st.divider()

        st.subheader("📡 Backend")

        if st.button(
            "Check backend",
            use_container_width=True,
        ):
            try:
                st.session_state.backend_status = (
                    client.health()
                )
            except APIClientError as exc:
                st.session_state.backend_status = {
                    "error": str(exc)
                }

        backend_status = (
            st.session_state.backend_status
        )

        if backend_status is None:
            st.info(
                "Backend status has not been checked."
            )

        elif "error" in backend_status:
            st.error(
                backend_status["error"]
            )

        else:
            st.success(
                "🟢 Backend reachable"
            )

        if api_key.strip():
            try:
                readiness = client.readiness(
                    api_key=api_key,
                )

                if readiness.ready:
                    st.success(
                        "🟢 RAG system ready"
                    )
                else:
                    st.warning(
                        "🟡 RAG system not ready"
                    )

                for name, ok in (
                    readiness.checks.items()
                ):
                    if ok:
                        st.caption(
                            f"✅ {name}"
                        )
                    else:
                        st.caption(
                            f"❌ {name}"
                        )

            except APIClientError as exc:
                st.warning(
                    f"Readiness check failed: {exc}"
                )

        # -----------------------------------------------------
        # Upload
        # -----------------------------------------------------

        st.divider()

        st.subheader("📄 Document Upload")

        upload_kb = st.selectbox(
            "Upload into",
            options=[
                (
                    "Level 1 · My Documents",
                    config.level1_knowledge_base_id,
                ),
                (
                    "Level 2 · Organization",
                    config.level2_knowledge_base_id,
                ),
            ],
            format_func=lambda item: item[0],
        )

        uploaded_file = st.file_uploader(
            "Choose a document",
            type=[
                "pdf",
                "txt",
                "docx",
            ],
        )

        upload_clicked = st.button(
            "Upload document",
            use_container_width=True,
            type="primary",
        )

        if upload_clicked:
            if not api_key.strip():
                st.error(
                    "Enter an API key before uploading."
                )
                return

            if uploaded_file is None:
                st.error(
                    "Choose a document first."
                )
                return

            with st.spinner(
                "Submitting document for ingestion..."
            ):
                try:
                    result = client.upload_document(
                        file_name=uploaded_file.name,
                        file_bytes=uploaded_file.getvalue(),
                        api_key=api_key,
                        content_type=uploaded_file.type,
                        knowledge_base_id=upload_kb[1],
                    )

                    st.success(
                        "Document ingestion started."
                    )

                    st.caption(
                        f"Job ID: `{result.job_id}`"
                    )

                    # Poll the background ingestion job.
                    if result.job_id:
                        progress = st.empty()

                        for _ in range(60):
                            job = (
                                client.get_ingestion_job(
                                    result.job_id,
                                    api_key,
                                )
                            )

                            progress.info(
                                f"Ingestion status: "
                                f"`{job.status}`"
                            )

                            if job.status in {
                                "completed",
                                "failed",
                            }:
                                break

                            time.sleep(1)

                        if job.status == "completed":
                            st.success(
                                "Document processed successfully."
                            )

                            metrics = []

                            if job.doc_id:
                                metrics.append(
                                    f"Document ID: `{job.doc_id}`"
                                )

                            if (
                                job.page_count
                                is not None
                            ):
                                metrics.append(
                                    f"Pages: {job.page_count}"
                                )

                            if (
                                job.chunk_count
                                is not None
                            ):
                                metrics.append(
                                    f"Chunks: {job.chunk_count}"
                                )

                            if metrics:
                                st.caption(
                                    " · ".join(metrics)
                                )

                        elif job.status == "failed":
                            st.error(
                                job.error_message
                                or "Document ingestion failed."
                            )

                except APIClientError as exc:
                    st.error(str(exc))

                except Exception as exc:
                    st.error(
                        f"Unexpected error: {exc}"
                    )

        st.divider()

        if st.button(
            "🗑️ Clear current results",
            use_container_width=True,
        ):
            st.session_state.level1_messages = []
            st.session_state.level2_results = None
            st.session_state.level3_results = None
            st.rerun()


# ============================================================
# Level 1
# ============================================================

def render_level1(
    client: RAGAPIClient,
) -> None:
    config = get_config()

    st.header(
        "Level 1 — Ask questions about my documents"
    )

    st.caption(
        "Upload documents into your personal/default "
        "knowledge base and ask natural-language questions."
    )

    for message in st.session_state.level1_messages:
        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )

            if (
                message["role"] == "assistant"
                and message.get("citations")
            ):
                render_citations(
                    message["citations"]
                )

    question = st.chat_input(
        "Ask a question about your documents...",
        key="level1_chat",
    )

    if not question:
        return

    api_key = st.session_state.api_key

    if not api_key.strip():
        st.error(
            "Enter your API key in the sidebar first."
        )
        return

    st.session_state.level1_messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Searching your documents and generating an answer..."
        ):
            try:
                result = client.query(
                    question=question,
                    knowledge_base_id=(
                        config.level1_knowledge_base_id
                    ),
                    api_key=api_key,
                )

                st.markdown(result.answer)

                render_citations(
                    result.citations
                )

                st.session_state.level1_messages.append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                        "citations": result.citations,
                    }
                )

            except APIClientError as exc:
                st.error(str(exc))

            except Exception as exc:
                st.error(
                    f"Unexpected error: {exc}"
                )


# ============================================================
# Level 2
# ============================================================

def render_level2(
    client: RAGAPIClient,
) -> None:
    config = get_config()

    st.header(
        "Level 2 — Search an organization's knowledge base"
    )

    st.caption(
        "Hybrid retrieval + CrossEncoder reranking. "
        "This mode returns ranked evidence rather than an LLM-generated answer."
    )

    query = st.text_area(
        "What are you looking for?",
        value=st.session_state.level2_query,
        placeholder=(
            "Example: What is the company's remote work policy?"
        ),
        height=100,
        key="level2_query_input",
    )

    controls = st.columns(2)

    with controls[0]:
        top_k = st.slider(
            "Results",
            min_value=1,
            max_value=20,
            value=10,
            key="level2_top_k",
        )

    with controls[1]:
        st.caption(
            f"Knowledge base: "
            f"`{config.level2_knowledge_base_id}`"
        )

    if st.button(
        "🔎 Search organization",
        type="primary",
        use_container_width=True,
        key="level2_search_button",
    ):
        if not st.session_state.api_key.strip():
            st.error(
                "Enter your API key in the sidebar first."
            )
            return

        if not query.strip():
            st.error(
                "Enter a search query."
            )
            return

        with st.spinner(
            "Searching, retrieving and reranking..."
        ):
            try:
                result = client.search(
                    query=query,
                    knowledge_base_id=(
                        config.level2_knowledge_base_id
                    ),
                    top_k=top_k,
                    api_key=st.session_state.api_key,
                )

                st.session_state.level2_query = query
                st.session_state.level2_results = result

            except APIClientError as exc:
                st.error(str(exc))

            except Exception as exc:
                st.error(
                    f"Unexpected error: {exc}"
                )

    result = st.session_state.level2_results

    if result is None:
        return

    st.divider()

    st.markdown(
        f"### {len(result.results)} ranked results"
    )

    if not result.results:
        st.info(
            "No relevant evidence was found."
        )
        return

    for index, item in enumerate(
        result.results,
        start=1,
    ):
        render_level2_result(
            index,
            item,
        )


# ============================================================
# Level 3
# ============================================================

def render_level3(
    client: RAGAPIClient,
) -> None:
    config = get_config()

    st.header(
        "Level 3 — Search and analyze resumes"
    )

    st.caption(
        "Natural-language recruiter search using "
        "structured candidate filters + hybrid retrieval + reranking."
    )

    query = st.text_area(
        "Describe the candidate you need",
        value=st.session_state.level3_query,
        placeholder=(
            "Example: Find Python developers with FastAPI "
            "backend experience"
        ),
        height=100,
        key="level3_query_input",
    )

    st.markdown("#### Candidate filters")

    row1 = st.columns(3)

    with row1[0]:
        skills_text = st.text_input(
            "Required skills",
            placeholder="python, fastapi",
            key="level3_skills",
        )

    with row1[1]:
        min_experience = st.number_input(
            "Minimum years",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.5,
            key="level3_min_experience",
        )

    with row1[2]:
        max_experience = st.number_input(
            "Maximum years",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.5,
            key="level3_max_experience",
        )

    row2 = st.columns(3)

    with row2[0]:
        location = st.text_input(
            "Location",
            placeholder="Kolkata",
            key="level3_location",
        )

    with row2[1]:
        title = st.text_input(
            "Current title",
            placeholder="Backend Engineer",
            key="level3_title",
        )

    with row2[2]:
        top_k = st.slider(
            "Candidates",
            min_value=1,
            max_value=20,
            value=10,
            key="level3_top_k",
        )

    st.caption(
        f"Knowledge base: "
        f"`{config.level3_knowledge_base_id}`"
    )

    if st.button(
        "👥 Find candidates",
        type="primary",
        use_container_width=True,
        key="level3_search_button",
    ):
        if not st.session_state.api_key.strip():
            st.error(
                "Enter your API key in the sidebar first."
            )
            return

        if not query.strip():
            st.error(
                "Describe the candidate you need."
            )
            return

        skills = [
            item.strip()
            for item in skills_text.split(",")
            if item.strip()
        ]

        min_years = (
            min_experience
            if min_experience > 0
            else None
        )

        max_years = (
            max_experience
            if max_experience > 0
            else None
        )

        if (
            min_years is not None
            and max_years is not None
            and max_years < min_years
        ):
            st.error(
                "Maximum experience cannot be less "
                "than minimum experience."
            )
            return

        with st.spinner(
            "Searching candidate profiles and resume evidence..."
        ):
            try:
                result = client.recruiter_search(
                    query=query,
                    knowledge_base_id=(
                        config.level3_knowledge_base_id
                    ),
                    top_k=top_k,
                    api_key=st.session_state.api_key,
                    skills=skills,
                    min_years_experience=min_years,
                    max_years_experience=max_years,
                    location=location or None,
                    title=title or None,
                )

                st.session_state.level3_query = query
                st.session_state.level3_results = result

            except APIClientError as exc:
                st.error(str(exc))

            except Exception as exc:
                st.error(
                    f"Unexpected error: {exc}"
                )

    result = st.session_state.level3_results

    if result is None:
        return

    st.divider()

    st.markdown(
        f"### {len(result.results)} candidates found"
    )

    if not result.results:
        st.info(
            "No candidates matched the supplied criteria."
        )
        return

    for index, candidate in enumerate(
        result.results,
        start=1,
    ):
        render_recruiter_candidate(
            index,
            candidate,
        )


# ============================================================
# Main application
# ============================================================

def main() -> None:
    initialize_session_state()

    client = get_api_client()

    render_sidebar(client)

    st.title("🔎 Production RAG Platform")

    st.caption(
        "One RAG core. Three application levels: "
        "document Q&A, enterprise knowledge search, "
        "and recruiter intelligence."
    )

    tabs = st.tabs(
        [
            "📄 Level 1 · My Documents",
            "🏢 Level 2 · Organization",
            "👥 Level 3 · Recruiter",
        ]
    )

    with tabs[0]:
        render_level1(client)

    with tabs[1]:
        render_level2(client)

    with tabs[2]:
        render_level3(client)


if __name__ == "__main__":
    main()

