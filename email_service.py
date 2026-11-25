import os
import re
import json
import time
import hmac
import hashlib
import secrets
import logging
import threading
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, Any

from flask import render_template

from config import Config
from database import (
    create_email_delivery,
    invalidate_old_tokens_for_screening,
    update_email_delivery_status,
    get_latest_email_delivery,
    get_email_delivery_by_token_hash,
    get_screening_by_uuid,
    get_email_diagnostics,
)

logger = logging.getLogger(__name__)

# Practical RFC-aligned email pattern (disallows whitespace, newlines, header injection chars)
_EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+$"
)

# In-memory lock & per-delivery active tracking to avoid duplicate concurrent dispatches
_DISPATCH_LOCK = threading.Lock()
_IN_FLIGHT_DELIVERIES = set()


@dataclass
class ProviderSendResult:
    accepted: bool
    retryable: bool = False
    provider_message_id: Optional[str] = None
    error_category: Optional[str] = None
    error_detail: Optional[str] = None
    status_code: Optional[int] = None
    retry_after_seconds: Optional[float] = None


def validate_and_normalize_email(raw_email: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Validates and normalizes a patient email address on the backend.
    Returns (normalized_email, error_code).
    If input is empty/whitespace, returns (None, None).
    """
    if raw_email is None:
        return None, None
    cleaned = str(raw_email).strip()
    if not cleaned:
        return None, None

    # Prevent CRLF header injection and oversized addresses
    if any(ch in cleaned for ch in ('\r', '\n', '\x00')) or len(cleaned) > 254:
        return None, 'invalid_email'

    if '..' in cleaned or not _EMAIL_REGEX.match(cleaned):
        return None, 'invalid_email'

    local_part, domain_part = cleaned.rsplit('@', 1)
    if len(local_part) > 64 or len(domain_part) > 253:
        return None, 'invalid_email'

    # Safe normalization: lowercase email for consistency
    return f"{local_part.lower()}@{domain_part.lower()}", None


def mask_email(email: Optional[str]) -> str:
    """Masks an email address for safe status display and privacy-preserving logs."""
    if not email or '@' not in email:
        return ''
    local, domain = email.split('@', 1)
    if len(local) <= 1:
        masked_local = '*'
    elif len(local) == 2:
        masked_local = local[0] + '*'
    else:
        masked_local = local[0] + ('*' * min(4, len(local) - 2)) + local[-1]
    return f"{masked_local}@{domain}"


def get_email_runtime_config() -> Dict[str, Any]:
    """
    Dynamically reads email configuration from environment variables at runtime
    with safe fallbacks to Config defaults.
    """
    enabled_raw = os.environ.get('EMAIL_REPORT_ENABLED')
    if enabled_raw is None:
        enabled = bool(getattr(Config, 'EMAIL_REPORT_ENABLED', True))
    else:
        enabled = str(enabled_raw).strip().lower() in ('1', 'true', 'yes', 'on')

    api_key = (os.environ.get('RESEND_API_KEY') if 'RESEND_API_KEY' in os.environ else getattr(Config, 'RESEND_API_KEY', '')) or ''
    mail_from = (os.environ.get('MAIL_FROM') if 'MAIL_FROM' in os.environ else getattr(Config, 'MAIL_FROM', '')) or ''
    app_base_url = (os.environ.get('APP_BASE_URL') if 'APP_BASE_URL' in os.environ else getattr(Config, 'APP_BASE_URL', 'https://ckd-screening-system.onrender.com')) or 'https://ckd-screening-system.onrender.com'

    expiry_raw = os.environ.get('DOWNLOAD_TOKEN_EXPIRY_HOURS')
    try:
        expiry_hours = int(expiry_raw) if expiry_raw is not None else int(getattr(Config, 'DOWNLOAD_TOKEN_EXPIRY_HOURS', 24))
    except (TypeError, ValueError):
        expiry_hours = 24
    expiry_hours = max(1, expiry_hours)

    return {
        'enabled': enabled,
        'api_key': api_key.strip(),
        'mail_from': mail_from.strip(),
        'app_base_url': app_base_url.strip().rstrip('/'),
        'expiry_hours': expiry_hours,
    }


def is_email_provider_configured(runtime_cfg: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, str]:
    """
    Checks whether transactional email delivery is enabled and properly configured.
    Returns (is_configured, reason_code, human_message).
    """
    cfg = runtime_cfg or get_email_runtime_config()
    if not cfg['enabled']:
        return False, 'email_disabled', 'Automatic email report delivery is currently disabled.'
    if not cfg['api_key']:
        return False, 'missing_api_key', 'Email delivery provider is not configured (missing RESEND_API_KEY).'
    if not cfg['mail_from']:
        return False, 'missing_sender', 'Email sender identity is not configured (missing MAIL_FROM).'
    return True, 'configured', 'Email provider configured.'


def generate_download_token(expiry_hours: Optional[int] = None) -> Tuple[str, str, datetime]:
    """
    Generates a cryptographically secure 256-bit download token,
    its SHA-256 hex digest (for database storage), and its UTC expiry datetime.
    """
    if expiry_hours is None:
        expiry_hours = get_email_runtime_config()['expiry_hours']
    raw_token = secrets.token_urlsafe(32)
    token_hash = hash_download_token(raw_token)
    expiry_dt = datetime.now(timezone.utc) + timedelta(hours=expiry_hours)
    return raw_token, token_hash, expiry_dt


def hash_download_token(raw_token: str) -> str:
    """Computes the SHA-256 hex digest of a raw download token."""
    return hashlib.sha256(raw_token.encode('utf-8')).hexdigest()


def verify_download_token(raw_token: str, db_path=None, db_url=None) -> Tuple[bool, str, Optional[Dict], Optional[Dict]]:
    """
    Verifies a download token cleanly:
    - Rejects malformed tokens
    - Looks up by SHA-256 token hash (never stores plaintext tokens)
    - Enforces UTC token expiry
    - Confirms screening record association
    Returns (is_valid, status_code, delivery_record, screening_record).
    status_code is one of: 'valid', 'invalid', 'expired'.
    """
    if not raw_token or not isinstance(raw_token, str) or len(raw_token) > 128:
        return False, 'invalid', None, None

    # Allow only URL-safe base64 characters
    if not re.match(r'^[A-Za-z0-9_-]+$', raw_token):
        return False, 'invalid', None, None

    token_hash = hash_download_token(raw_token)
    delivery = get_email_delivery_by_token_hash(token_hash, db_path=db_path, db_url=db_url)
    if not delivery:
        return False, 'invalid', None, None

    stored_hash = delivery.get('token_hash') or ''
    if not hmac.compare_digest(stored_hash, token_hash):
        return False, 'invalid', None, None

    expiry_dt = delivery.get('token_expiry_dt')
    if expiry_dt is None or datetime.now(timezone.utc) > expiry_dt:
        return False, 'expired', delivery, None

    screening = get_screening_by_uuid(delivery['screening_uuid'], db_path=db_path, db_url=db_url)
    if not screening or screening.get('screening_uuid') != delivery.get('screening_uuid'):
        return False, 'invalid', delivery, None

    return True, 'valid', delivery, screening


def build_https_download_url(raw_token: str, runtime_cfg: Optional[Dict[str, Any]] = None) -> str:
    """
    Constructs an HTTPS public download link containing ONLY the opaque token.
    Never exposes patient IDs, clinical values, or internal filesystem paths.
    """
    cfg = runtime_cfg or get_email_runtime_config()
    base_url = cfg.get('app_base_url') or 'https://ckd-screening-system.onrender.com'
    base_url = base_url.strip().rstrip('/')

    # Enforce HTTPS for all public download links
    if base_url.startswith('http://'):
        base_url = 'https://' + base_url[len('http://'):]
    elif not base_url.startswith('https://'):
        base_url = 'https://' + base_url.lstrip('/')

    return f"{base_url}/report/download/{raw_token}"


class BaseEmailProvider:
    """Modular base interface for transactional email providers (e.g. Resend, Brevo)."""
    provider_name = 'base'

    def send_email(
        self,
        recipient_email: str,
        subject: str,
        html_body: str,
        text_body: str,
        idempotency_key: Optional[str] = None,
    ) -> ProviderSendResult:
        raise NotImplementedError


class ResendEmailProvider(BaseEmailProvider):
    """Transactional email integration via Resend REST API v1."""
    provider_name = 'resend'
    API_ENDPOINT = 'https://api.resend.com/emails'

    def __init__(self, api_key: str, mail_from: str, timeout_seconds: int = 10):
        self.api_key = api_key
        self.mail_from = mail_from
        self.timeout_seconds = timeout_seconds

    def send_email(
        self,
        recipient_email: str,
        subject: str,
        html_body: str,
        text_body: str,
        idempotency_key: Optional[str] = None,
    ) -> ProviderSendResult:
        payload = json.dumps({
            'from': self.mail_from,
            'to': [recipient_email],
            'subject': subject,
            'html': html_body,
            'text': text_body,
        }).encode('utf-8')

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'RenalCare-Clinical-Screening/1.1',
        }
        if idempotency_key:
            headers['Idempotency-Key'] = idempotency_key

        req = urllib.request.Request(self.API_ENDPOINT, data=payload, headers=headers, method='POST')

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                status_code = getattr(resp, 'status', 200)
                raw_body = resp.read().decode('utf-8', errors='replace')
                try:
                    data = json.loads(raw_body) if raw_body else {}
                except json.JSONDecodeError:
                    data = {}

                if 200 <= status_code < 300:
                    return ProviderSendResult(
                        accepted=True,
                        retryable=False,
                        provider_message_id=data.get('id'),
                        status_code=status_code,
                    )
                return ProviderSendResult(
                    accepted=False,
                    retryable=status_code >= 500,
                    error_category='unexpected_response',
                    error_detail=f'Unexpected HTTP {status_code} response from email provider.',
                    status_code=status_code,
                )

        except urllib.error.HTTPError as http_err:
            status_code = http_err.code
            raw_err = ''
            try:
                raw_err = http_err.read().decode('utf-8', errors='replace')
                err_json = json.loads(raw_err) if raw_err else {}
            except Exception:
                err_json = {}

            err_name = str(err_json.get('name') or '').strip().lower()
            err_msg = str(err_json.get('message') or http_err.reason or 'Provider HTTP error').strip()

            # Classify retryable vs permanent failures accurately
            if status_code == 429:
                retry_after = None
                try:
                    ra_header = http_err.headers.get('retry-after') if http_err.headers else None
                    if ra_header is not None:
                        retry_after = min(5.0, max(0.5, float(ra_header)))
                except Exception:
                    retry_after = None

                if err_name == 'rate_limit_exceeded':
                    return ProviderSendResult(
                        accepted=False,
                        retryable=True,
                        error_category='rate_limit_exceeded',
                        error_detail='Email provider temporary rate limit reached.',
                        status_code=429,
                        retry_after_seconds=retry_after,
                    )
                # Daily/monthly quota exhaustion should NOT spin retries
                return ProviderSendResult(
                    accepted=False,
                    retryable=False,
                    error_category=err_name or 'quota_exceeded',
                    error_detail='Email provider sending quota reached.',
                    status_code=429,
                )

            if status_code == 401:
                return ProviderSendResult(
                    accepted=False,
                    retryable=False,
                    error_category='invalid_api_key',
                    error_detail='Email provider rejected API key credentials.',
                    status_code=401,
                )

            if status_code == 403:
                detail = f'Provider restricted delivery (HTTP 403): {err_msg}' if err_msg else 'Sender/recipient restriction on email provider (verify custom domain or test recipient in Resend).'
                return ProviderSendResult(
                    accepted=False,
                    retryable=False,
                    error_category='domain_verification_required',
                    error_detail=detail[:250],
                    status_code=403,
                )

            if status_code in (400, 422):
                return ProviderSendResult(
                    accepted=False,
                    retryable=False,
                    error_category=err_name or 'validation_error',
                    error_detail=f'Email request validation error: {err_msg[:180]}',
                    status_code=status_code,
                )

            if status_code >= 500:
                return ProviderSendResult(
                    accepted=False,
                    retryable=True,
                    error_category='provider_server_error',
                    error_detail=f'Email provider temporary server error (HTTP {status_code}).',
                    status_code=status_code,
                )

            return ProviderSendResult(
                accepted=False,
                retryable=False,
                error_category='provider_rejected',
                error_detail=f'Email provider error (HTTP {status_code}).',
                status_code=status_code,
            )

        except (urllib.error.URLError, TimeoutError, OSError) as net_err:
            return ProviderSendResult(
                accepted=False,
                retryable=True,
                error_category='network_error',
                error_detail=f'Temporary network timeout reaching email provider ({type(net_err).__name__}).',
            )
        except Exception as exc:
            return ProviderSendResult(
                accepted=False,
                retryable=False,
                error_category='internal_error',
                error_detail=f'Unexpected email dispatch error ({type(exc).__name__}).',
            )


def render_report_email_bodies(patient_name: Optional[str], screening_date: str, download_url: str, expiry_hours: int) -> Tuple[str, str, str]:
    """
    Renders non-sensitive HTML and plain-text email templates.
    NEVER includes clinical findings, lab parameters, or risk predictions.
    """
    subject = "Your RenalCare Screening Report Is Ready"
    greeting_name = patient_name.strip() if (patient_name and isinstance(patient_name, str) and patient_name.strip()) else None

    context = {
        'patient_name': greeting_name,
        'screening_date': screening_date,
        'download_url': download_url,
        'expiry_hours': expiry_hours,
    }
    html_body = render_template('emails/report_ready.html', **context)
    text_body = render_template('emails/report_ready.txt', **context)
    return subject, html_body, text_body


def deliver_email_with_retries(
    delivery_id: int,
    screening_uuid: str,
    recipient_email: str,
    subject: str,
    html_body: str,
    text_body: str,
    idempotency_key: str,
    provider: BaseEmailProvider,
    max_attempts: int = 3,
    retry_delays: Tuple[float, ...] = (1.0, 2.0),
    db_path=None,
    db_url=None,
) -> ProviderSendResult:
    """
    Executes email delivery with bounded exponential backoff retries on temporary errors
    and strict duplicate-send prevention.
    """
    with _DISPATCH_LOCK:
        if delivery_id in _IN_FLIGHT_DELIVERIES:
            return ProviderSendResult(
                accepted=False,
                retryable=False,
                error_category='duplicate_prevented',
                error_detail='Email delivery already in progress.',
            )
        _IN_FLIGHT_DELIVERIES.add(delivery_id)

    try:
        # Guard against duplicate sends if already accepted for this screening delivery row
        latest = get_latest_email_delivery(screening_uuid, db_path=db_path, db_url=db_url)
        if latest and latest.get('id') == delivery_id and latest.get('delivery_status') == 'sent':
            return ProviderSendResult(
                accepted=True,
                provider_message_id=latest.get('provider_message_id'),
            )

        last_result = ProviderSendResult(accepted=False, error_category='unknown', error_detail='No attempt made')

        for attempt in range(1, max_attempts + 1):
            now_utc = datetime.now(timezone.utc)
            last_result = provider.send_email(
                recipient_email=recipient_email,
                subject=subject,
                html_body=html_body,
                text_body=text_body,
                idempotency_key=idempotency_key,
            )

            if last_result.accepted:
                update_email_delivery_status(
                    delivery_id=delivery_id,
                    delivery_status='sent',
                    provider_message_id=last_result.provider_message_id,
                    attempts=attempt,
                    sent_at=now_utc,
                    last_attempt_at=now_utc,
                    error_category='',
                    error_detail='',
                    db_path=db_path,
                    db_url=db_url,
                )
                logger.info(
                    f"Email accepted by provider: screening={screening_uuid} delivery_id={delivery_id} "
                    f"recipient={mask_email(recipient_email)} attempts={attempt}"
                )
                return last_result

            # Permanent error or exhausted retries -> mark failed safely
            if not last_result.retryable or attempt >= max_attempts:
                update_email_delivery_status(
                    delivery_id=delivery_id,
                    delivery_status='failed',
                    attempts=attempt,
                    last_attempt_at=now_utc,
                    error_category=last_result.error_category or 'delivery_failed',
                    error_detail=last_result.error_detail or 'Email delivery failed.',
                    db_path=db_path,
                    db_url=db_url,
                )
                logger.warning(
                    f"Email delivery failed: screening={screening_uuid} delivery_id={delivery_id} "
                    f"recipient={mask_email(recipient_email)} category={last_result.error_category} attempts={attempt}"
                )
                return last_result

            # Temporary error with remaining retry budget
            update_email_delivery_status(
                delivery_id=delivery_id,
                delivery_status='pending',
                attempts=attempt,
                last_attempt_at=now_utc,
                error_category=last_result.error_category,
                error_detail=f"Attempt {attempt} encountered temporary issue; retrying...",
                db_path=db_path,
                db_url=db_url,
            )
            delay = last_result.retry_after_seconds
            if delay is None:
                delay_idx = min(attempt - 1, len(retry_delays) - 1)
                delay = retry_delays[delay_idx] if retry_delays else 0.0
            if delay > 0:
                time.sleep(delay)

        return last_result
    finally:
        with _DISPATCH_LOCK:
            _IN_FLIGHT_DELIVERIES.discard(delivery_id)


def _dispatch_delivery(
    app,
    delivery_id: int,
    screening_uuid: str,
    recipient_email: str,
    subject: str,
    html_body: str,
    text_body: str,
    idempotency_key: str,
    provider: BaseEmailProvider,
    background: bool = True,
    retry_delays: Tuple[float, ...] = (1.0, 2.0),
    db_path=None,
    db_url=None,
):
    if not background:
        return deliver_email_with_retries(
            delivery_id=delivery_id,
            screening_uuid=screening_uuid,
            recipient_email=recipient_email,
            subject=subject,
            html_body=html_body,
            text_body=text_body,
            idempotency_key=idempotency_key,
            provider=provider,
            retry_delays=retry_delays,
            db_path=db_path,
            db_url=db_url,
        )

    def _worker():
        with app.app_context():
            deliver_email_with_retries(
                delivery_id=delivery_id,
                screening_uuid=screening_uuid,
                recipient_email=recipient_email,
                subject=subject,
                html_body=html_body,
                text_body=text_body,
                idempotency_key=idempotency_key,
                provider=provider,
                retry_delays=retry_delays,
                db_path=db_path,
                db_url=db_url,
            )

    t = threading.Thread(target=_worker, daemon=True, name=f"renalcare-email-{delivery_id}")
    t.start()
    return None


def initiate_screening_email_workflow(
    app,
    screening_record: Dict[str, Any],
    raw_email: Optional[str],
    consent_given: bool,
    background: bool = True,
    provider: Optional[BaseEmailProvider] = None,
    retry_delays: Tuple[float, ...] = (1.0, 2.0),
    db_path=None,
    db_url=None,
) -> Dict[str, Any]:
    """
    Orchestrates post-screening email workflow safely:
    1. Validates & normalizes optional email.
    2. Enforces explicit patient consent.
    3. Verifies email provider configuration gracefully.
    4. Issues cryptographic download token (SHA-256 hash stored, 24h expiry).
    5. Renders email templates & dispatches without blocking screening flow.
    """
    screening_uuid = screening_record['screening_uuid']
    patient_id = screening_record.get('patient_id')
    normalized_email, email_err = validate_and_normalize_email(raw_email)

    # Case 1: No email provided
    if normalized_email is None and email_err is None:
        delivery_id = create_email_delivery(
            screening_uuid=screening_uuid,
            patient_id=patient_id,
            recipient_email=None,
            consent_recorded=False,
            delivery_status='not_provided',
            db_path=db_path,
            db_url=db_url,
        )
        return {'delivery_id': delivery_id, 'delivery_status': 'not_provided'}

    # Case 2: Invalid email format (never blocks screening)
    if email_err == 'invalid_email':
        delivery_id = create_email_delivery(
            screening_uuid=screening_uuid,
            patient_id=patient_id,
            recipient_email=None,
            consent_recorded=bool(consent_given),
            delivery_status='failed',
            error_category='invalid_email',
            error_detail='The supplied email address format was invalid; email was not sent.',
            db_path=db_path,
            db_url=db_url,
        )
        return {'delivery_id': delivery_id, 'delivery_status': 'failed', 'error_category': 'invalid_email'}

    # Case 3: Valid email supplied, but patient consent was NOT provided
    if not consent_given:
        delivery_id = create_email_delivery(
            screening_uuid=screening_uuid,
            patient_id=patient_id,
            recipient_email=normalized_email,
            consent_recorded=False,
            delivery_status='consent_not_provided',
            error_category='consent_required',
            error_detail='Patient consent checkbox was not checked; email was not sent.',
            db_path=db_path,
            db_url=db_url,
        )
        return {'delivery_id': delivery_id, 'delivery_status': 'consent_not_provided'}

    # Case 4: Email & consent valid — check provider configuration
    cfg = get_email_runtime_config()
    configured, reason_code, reason_msg = is_email_provider_configured(cfg)
    if not configured and provider is None:
        delivery_id = create_email_delivery(
            screening_uuid=screening_uuid,
            patient_id=patient_id,
            recipient_email=normalized_email,
            consent_recorded=True,
            delivery_status='not_configured',
            error_category=reason_code,
            error_detail=reason_msg,
            db_path=db_path,
            db_url=db_url,
        )
        return {'delivery_id': delivery_id, 'delivery_status': 'not_configured', 'error_category': reason_code}

    # Case 5: Ready to generate token & send email
    raw_token, token_hash, expiry_dt = generate_download_token(cfg['expiry_hours'])
    download_url = build_https_download_url(raw_token, cfg)
    idempotency_key = f"renalcare-{screening_uuid}-{secrets.token_hex(6)}"

    active_provider = provider or ResendEmailProvider(
        api_key=cfg['api_key'],
        mail_from=cfg['mail_from'],
    )

    delivery_id = create_email_delivery(
        screening_uuid=screening_uuid,
        patient_id=patient_id,
        recipient_email=normalized_email,
        consent_recorded=True,
        delivery_status='pending',
        provider_name=active_provider.provider_name,
        idempotency_key=idempotency_key,
        token_hash=token_hash,
        token_expiry=expiry_dt,
        db_path=db_path,
        db_url=db_url,
    )
    invalidate_old_tokens_for_screening(screening_uuid, exclude_delivery_id=delivery_id, db_path=db_path, db_url=db_url)

    subject, html_body, text_body = render_report_email_bodies(
        patient_name=screening_record.get('patient_name'),
        screening_date=screening_record.get('formatted_date') or datetime.now(timezone.utc).strftime('%b %d, %Y'),
        download_url=download_url,
        expiry_hours=cfg['expiry_hours'],
    )

    _dispatch_delivery(
        app=app,
        delivery_id=delivery_id,
        screening_uuid=screening_uuid,
        recipient_email=normalized_email,
        subject=subject,
        html_body=html_body,
        text_body=text_body,
        idempotency_key=idempotency_key,
        provider=active_provider,
        background=background,
        retry_delays=retry_delays,
        db_path=db_path,
        db_url=db_url,
    )

    latest = get_latest_email_delivery(screening_uuid, db_path=db_path, db_url=db_url)
    return {
        'delivery_id': delivery_id,
        'delivery_status': latest['delivery_status'] if latest else 'pending',
        'raw_token': raw_token if app.config.get('TESTING') else None,
    }


def resend_screening_email(
    app,
    screening_record: Dict[str, Any],
    cooldown_seconds: int = 15,
    background: bool = True,
    provider: Optional[BaseEmailProvider] = None,
    retry_delays: Tuple[float, ...] = (1.0, 2.0),
    db_path=None,
    db_url=None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Allows an authorized healthcare worker to issue a fresh 24-hour download link
    and resend the email for a screening record that already has recorded consent.
    Prevents rapid accidental duplicate submissions.
    """
    screening_uuid = screening_record['screening_uuid']
    latest = get_latest_email_delivery(screening_uuid, db_path=db_path, db_url=db_url)

    if not latest or not latest.get('recipient_email'):
        return False, "No valid patient email address is recorded for this screening.", {'delivery_status': 'not_provided'}

    if not latest.get('consent_recorded'):
        return False, "Cannot resend report: patient consent was not recorded.", {'delivery_status': 'consent_not_provided'}

    if latest.get('delivery_status') == 'pending':
        return False, "Email delivery is already in progress. Please wait before retrying.", {'delivery_status': 'pending'}

    # Duplicate click protection during short cooldown window after a successful send
    if latest.get('delivery_status') == 'sent' and latest.get('sent_at_dt') and cooldown_seconds > 0:
        elapsed = (datetime.now(timezone.utc) - latest['sent_at_dt']).total_seconds()
        if elapsed < cooldown_seconds:
            return (
                False,
                f"Report was just emailed {int(elapsed)}s ago. Please wait {int(cooldown_seconds - elapsed) + 1}s before resending.",
                {'delivery_status': 'sent', 'duplicate_prevented': True},
            )

    cfg = get_email_runtime_config()
    configured, reason_code, reason_msg = is_email_provider_configured(cfg)
    if not configured and provider is None:
        return False, reason_msg, {'delivery_status': 'not_configured', 'error_category': reason_code}

    raw_token, token_hash, expiry_dt = generate_download_token(cfg['expiry_hours'])
    download_url = build_https_download_url(raw_token, cfg)
    idempotency_key = f"renalcare-resend-{screening_uuid}-{secrets.token_hex(6)}"

    active_provider = provider or ResendEmailProvider(
        api_key=cfg['api_key'],
        mail_from=cfg['mail_from'],
    )

    delivery_id = create_email_delivery(
        screening_uuid=screening_uuid,
        patient_id=screening_record.get('patient_id') or latest.get('patient_id'),
        recipient_email=latest['recipient_email'],
        consent_recorded=True,
        delivery_status='pending',
        provider_name=active_provider.provider_name,
        idempotency_key=idempotency_key,
        token_hash=token_hash,
        token_expiry=expiry_dt,
        db_path=db_path,
        db_url=db_url,
    )
    invalidate_old_tokens_for_screening(screening_uuid, exclude_delivery_id=delivery_id, db_path=db_path, db_url=db_url)

    subject, html_body, text_body = render_report_email_bodies(
        patient_name=screening_record.get('patient_name'),
        screening_date=screening_record.get('formatted_date') or datetime.now(timezone.utc).strftime('%b %d, %Y'),
        download_url=download_url,
        expiry_hours=cfg['expiry_hours'],
    )

    _dispatch_delivery(
        app=app,
        delivery_id=delivery_id,
        screening_uuid=screening_uuid,
        recipient_email=latest['recipient_email'],
        subject=subject,
        html_body=html_body,
        text_body=text_body,
        idempotency_key=idempotency_key,
        provider=active_provider,
        background=background,
        retry_delays=retry_delays,
        db_path=db_path,
        db_url=db_url,
    )

    updated = get_latest_email_delivery(screening_uuid, db_path=db_path, db_url=db_url)
    final_status = updated['delivery_status'] if updated else 'pending'
    if final_status == 'sent':
        msg = "A new secure download link has been emailed to the patient."
    elif final_status == 'failed':
        msg = updated.get('error_detail') or "Email delivery failed when contacting the provider."
    else:
        msg = "A replacement download link has been queued for email delivery."

    return final_status in ('sent', 'pending'), msg, {
        'delivery_id': delivery_id,
        'delivery_status': final_status,
        'raw_token': raw_token if app.config.get('TESTING') else None,
    }


def format_delivery_status_summary(delivery: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Formats a clean, privacy-safe summary dictionary for templates and `/api/email-status`.
    Never includes raw recipient email or clinical metrics.
    """
    if not delivery:
        return {
            'delivery_status': 'not_provided',
            'badge_label': 'Email Not Provided',
            'message': 'No patient email address was supplied for this screening.',
            'can_resend': False,
            'masked_email': '',
            'attempts': 0,
            'consent_recorded': False,
        }

    status = delivery.get('delivery_status', 'not_provided')
    masked = mask_email(delivery.get('recipient_email'))
    consent = bool(delivery.get('consent_recorded'))
    attempts = int(delivery.get('attempts') or 0)
    err_detail = delivery.get('error_detail') or ''

    if status == 'sent':
        return {
            'delivery_status': 'sent',
            'badge_label': 'Email Sent',
            'message': f"Your screening report has been emailed successfully to {masked}.",
            'can_resend': consent and bool(masked),
            'masked_email': masked,
            'attempts': attempts,
            'sent_at': delivery.get('sent_at'),
            'token_expiry': delivery.get('token_expiry'),
            'consent_recorded': consent,
        }
    if status == 'pending':
        return {
            'delivery_status': 'pending',
            'badge_label': 'Delivery Pending',
            'message': f"Your report has been generated. Email delivery to {masked} is pending.",
            'can_resend': False,
            'masked_email': masked,
            'attempts': attempts,
            'consent_recorded': consent,
        }
    if status == 'failed':
        base_msg = "Your report is ready, but the email could not be sent."
        if err_detail:
            base_msg = f"{base_msg} ({err_detail})"
        return {
            'delivery_status': 'failed',
            'badge_label': 'Delivery Failed',
            'message': base_msg,
            'can_resend': consent and bool(masked),
            'masked_email': masked,
            'attempts': attempts,
            'error_category': delivery.get('error_category'),
            'consent_recorded': consent,
        }
    if status == 'not_configured':
        return {
            'delivery_status': 'not_configured',
            'badge_label': 'Email Not Configured',
            'message': (
                f"Report generated. Automatic email delivery to {masked} could not run because the email "
                "provider (RESEND_API_KEY / MAIL_FROM) is not configured on this server."
            ),
            'can_resend': consent and bool(masked),
            'masked_email': masked,
            'attempts': attempts,
            'error_category': delivery.get('error_category'),
            'consent_recorded': consent,
        }
    if status == 'consent_not_provided':
        return {
            'delivery_status': 'consent_not_provided',
            'badge_label': 'Consent Not Provided',
            'message': (
                "Report generated. Email was not sent because explicit patient consent was not provided."
            ),
            'can_resend': False,
            'masked_email': masked,
            'attempts': 0,
            'consent_recorded': False,
        }

    return {
        'delivery_status': 'not_provided',
        'badge_label': 'Email Not Provided',
        'message': 'No patient email address was supplied for this screening.',
        'can_resend': False,
        'masked_email': '',
        'attempts': 0,
        'consent_recorded': False,
    }


def get_email_service_diagnostics(limit: int = 50, db_path=None, db_url=None) -> Dict[str, Any]:
    """
    Returns diagnostic summary of transactional email delivery status, provider configuration,
    and sanitized recent delivery records for authorized clinician review.
    """
    cfg = get_email_runtime_config()
    is_conf, code, msg = is_email_provider_configured(cfg)
    deliveries = get_email_diagnostics(limit=limit, db_path=db_path, db_url=db_url)

    sender = cfg.get('mail_from', '')
    is_resend_dev = 'onboarding@resend.dev' in sender.lower()

    sandbox_note = (
        "On Resend free tier using onboarding@resend.dev, emails can only be delivered to your registered Resend account address. "
        "To deliver reports to arbitrary patient emails, configure a custom domain with SPF/DKIM records in Resend and update MAIL_FROM."
        if is_resend_dev else
        "Custom sender domain configured."
    )

    return {
        'provider_configured': is_conf,
        'configuration_code': code,
        'configuration_message': msg,
        'enabled': cfg['enabled'],
        'mail_from': cfg['mail_from'],
        'app_base_url': cfg['app_base_url'],
        'expiry_hours': cfg['expiry_hours'],
        'sandbox_restriction_note': sandbox_note,
        'recent_deliveries': deliveries,
    }
