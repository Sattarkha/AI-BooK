from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

# Request/Response models
class ValidationRequest(BaseModel):
    content: str
    content_type: str  # e.g., "textbook", "exercise", "code"


class ValidationResponse(BaseModel):
    is_valid: bool
    issues: list
    suggestions: list


@router.post("/validate", response_model=ValidationResponse)
async def validate_content(request: ValidationRequest):
    """
    Validate content for the textbook
    """
    # In a real implementation, this would perform content validation
    # For now, return mock validation results
    is_valid = len(request.content) > 10  # Simple validation: content must be longer than 10 chars
    issues = [] if is_valid else ["Content is too short"]
    suggestions = ["Add more detailed explanations"] if is_valid else []

    return ValidationResponse(
        is_valid=is_valid,
        issues=issues,
        suggestions=suggestions
    )


@router.get("/health")
async def utils_health():
    """
    Check the health of the utils service
    """
    return {"status": "healthy", "service": "utils"}


class SanitizeRequest(BaseModel):
    text: str


class SanitizeResponse(BaseModel):
    sanitized_text: str


@router.post("/sanitize", response_model=SanitizeResponse)
async def sanitize_text(request: SanitizeRequest):
    """
    Sanitize user input to prevent XSS and other security issues
    """
    # In a real implementation, this would sanitize the input properly
    # For now, return the same text (in a real app, use proper sanitization)
    sanitized = request.text  # Placeholder - implement proper sanitization

    return SanitizeResponse(sanitized_text=sanitized)