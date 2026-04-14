from __future__ import annotations

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.errors import AppError
from app.pipeline import MedicalPipeline
from app.schemas import ErrorResponse, HealthResponse, RateStatusResponse
from app.services.rate_limiter import RateLimiter
from app.services.runtime_judge import RuntimeJudgeService
from app.tools.openfda import OpenFDATool
from app.utils.extractor import extract_text_from_pdf_bytes, validate_raw_text

settings = get_settings()
rate_limiter = RateLimiter(
    daily_limit=settings.daily_limit,
    rest_url=settings.upstash_redis_rest_url,
    rest_token=settings.upstash_redis_rest_token,
)
openfda_tool = OpenFDATool()
pipeline = MedicalPipeline(settings, openfda_tool)
runtime_judge = RuntimeJudgeService(settings)

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AppError)
async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(code=exc.code, message=exc.message, details=exc.details).model_dump(),
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            code="internal_error",
            message="The server could not complete the analysis request.",
            details={"error": str(exc)},
        ).model_dump(),
    )


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    redis_health = await rate_limiter.healthcheck()
    retrieval_health = pipeline.medication_rag.healthcheck()
    return HealthResponse(
        status="ok",
        dependencies={
            "redis": {"status": redis_health["status"], "details": redis_health},
            "openai": {
                "status": "configured" if settings.openai_enabled else "fallback",
                "details": {
                    "classifier_model": settings.openai_classifier_model,
                    "analyst_model": settings.openai_analyst_model,
                    "embedding_model": settings.openai_embedding_model,
                },
            },
            "openfda": {"status": "available", "details": {"base_url": openfda_tool.base_url}},
            "retrieval": {"status": retrieval_health["status"], "details": retrieval_health},
        },
    )


@app.get("/api/rate-status", response_model=RateStatusResponse)
async def rate_status(request: Request) -> RateStatusResponse:
    status = await rate_limiter.peek(_client_ip(request))
    return RateStatusResponse(
        remaining=status.remaining,
        daily_limit=status.daily_limit,
        reset_at=status.reset_at.isoformat(),
    )


@app.post("/api/analyze")
async def analyze(
    request: Request,
    file: UploadFile | None = File(default=None),
    raw_text: str | None = Form(default=None),
):
    if file and raw_text:
        raise AppError(status_code=400, code="multiple_inputs", message="Submit either a PDF file or raw_text, not both.")
    if not file and raw_text is None:
        raise AppError(status_code=400, code="missing_input", message="Provide either a PDF file or raw_text.")

    partial_data_reasons: list[str] = []
    if file:
        filename = file.filename or ""
        if file.content_type not in {"application/pdf", "application/octet-stream"} and not filename.lower().endswith(".pdf"):
            raise AppError(status_code=415, code="unsupported_file_type", message="Only PDF uploads are supported in this MVP.")
        file_bytes = await file.read()
        if not file_bytes:
            raise AppError(status_code=400, code="empty_file", message="The uploaded file was empty.")
        if len(file_bytes) > settings.max_file_size_bytes:
            raise AppError(
                status_code=413,
                code="file_too_large",
                message="The uploaded file exceeds the maximum supported size for this demo.",
                details={
                    "max_file_size_kb": settings.max_file_size_kb,
                    "max_file_size_bytes": settings.max_file_size_bytes,
                },
            )
        extracted = extract_text_from_pdf_bytes(file_bytes, settings.request_char_limit)
        if extracted.truncated:
            partial_data_reasons.append("The PDF text was truncated to fit the demo processing limit.")
    else:
        extracted = validate_raw_text(raw_text, settings.request_char_limit)

    limit_status = await rate_limiter.consume(_client_ip(request))
    if limit_status.limit_exceeded:
        raise AppError(
            status_code=429,
            code="rate_limit_exceeded",
            message="You have used all free analyses for today.",
            details={
                "daily_limit": limit_status.daily_limit,
                "reset_at": limit_status.reset_at.isoformat(),
            },
        )

    result = await pipeline.analyze(
        text=extracted.text,
        rate_limit_remaining=limit_status.remaining,
        daily_limit=limit_status.daily_limit,
        reset_at=limit_status.reset_at.isoformat(),
        partial_data_reasons=partial_data_reasons,
    )
    result = await runtime_judge.review(report_text=extracted.text, analysis=result)
    return result.model_dump()


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
