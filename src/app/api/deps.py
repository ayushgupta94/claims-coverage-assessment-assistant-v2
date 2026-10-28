"""FastAPI dependencies. The object graph is built once in main.create_app()
and stored on app.state; routes just ask for it here. Tests can override
any of these with app.dependency_overrides."""

from __future__ import annotations

from fastapi import Request

from app.config import Settings
from app.services.claim_assessment_service import ClaimAssessmentService


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_claim_service(request: Request) -> ClaimAssessmentService:
    return request.app.state.claim_service
