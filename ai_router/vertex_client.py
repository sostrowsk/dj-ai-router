import logging
from pathlib import Path

from django.conf import settings
from google import genai
from google.auth import default as google_auth_default
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

VERTEX_MODEL_CONFIG = {
    "gemini-3.1-pro-preview": {"supports_temp": True, "engine": "gemini"},
}


def _get_gemini_credentials():
    """Load credentials for Gemini — separate service account or ADC fallback."""
    creds_path = settings.GCP_GEMINI_CREDENTIALS
    if creds_path:
        resolved = Path(settings.BASE_DIR) / creds_path
        if resolved.exists():
            return service_account.Credentials.from_service_account_file(
                str(resolved),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        logger.warning("GCP_GEMINI_CREDENTIALS path not found: %s", resolved)
    creds, _ = google_auth_default()
    return creds


def _get_gemini_client():
    """Create a google.genai Client — prefers Developer API (API key), falls back to Vertex AI."""
    api_key = settings.GCP_GEMINI_API_KEY
    if api_key:
        logger.info("Using Gemini Developer API (API key)")
        return genai.Client(api_key=api_key)
    logger.info("Using Gemini via Vertex AI (service account)")
    return genai.Client(
        vertexai=True,
        project=settings.GCP_GEMINI_PROJECT_ID,
        location=settings.GCP_GEMINI_REGION,
        credentials=_get_gemini_credentials(),
    )
