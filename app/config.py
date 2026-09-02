import os
from dotenv import load_dotenv

load_dotenv()

KITE_API_KEY = os.getenv("KITE_API_KEY", "").strip()
KITE_API_SECRET = os.getenv("KITE_API_SECRET", "").strip()
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "").strip()
KITE_REDIRECT_URL = os.getenv("KITE_REDIRECT_URL", "http://127.0.0.1:8000/callback").strip()


def validate_credentials(require_access_token: bool = False) -> None:
    missing = []
    if not KITE_API_KEY:
        missing.append("KITE_API_KEY")
    if not KITE_API_SECRET:
        missing.append("KITE_API_SECRET")
    if require_access_token and not KITE_ACCESS_TOKEN:
        missing.append("KITE_ACCESS_TOKEN")
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
