from kiteconnect import KiteConnect
from .config import KITE_API_KEY, KITE_API_SECRET, KITE_ACCESS_TOKEN, validate_credentials


def get_kite(require_access_token: bool = False) -> KiteConnect:
    validate_credentials(require_access_token=require_access_token)
    kite = KiteConnect(api_key=KITE_API_KEY)
    if KITE_ACCESS_TOKEN:
        kite.set_access_token(KITE_ACCESS_TOKEN)
    return kite


def login_url() -> str:
    return get_kite(require_access_token=False).login_url()


def generate_access_token(request_token: str) -> dict:
    validate_credentials(require_access_token=False)
    kite = KiteConnect(api_key=KITE_API_KEY)
    session = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
    return session


def profile() -> dict:
    kite = get_kite(require_access_token=True)
    return kite.profile()
