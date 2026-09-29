"""
Helper to release brakes and activate FCI from code at SPARK startup.

Wraps franky.RobotWebSession (the Python client for the same Desk REST
API that the web UI uses) so the SPARK server can come up from cold
without anyone touching the Desk page. Credentials are loaded from
either env vars (FRANKA_DESK_USER + FRANKA_DESK_PASSWORD) or a JSON
file at ~/.franka_desk.json. Failures degrade gracefully: we log and
let the franky FCI constructor do its own thing afterwards.

File format:
    {"user": "your-desk-user", "password": "your-desk-pass"}
Permissions:
    chmod 600 ~/.franka_desk.json
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import franky
except ImportError:
    franky = None

_DEFAULT_CREDS_PATH = pathlib.Path("~/.franka_desk.json").expanduser()


def _load_credentials(
    creds_path: Optional[pathlib.Path] = None,
) -> Optional[Tuple[str, str]]:
    # Return (user, password) from env vars or JSON file, or None.
    env_user = os.environ.get("FRANKA_DESK_USER")
    env_pass = os.environ.get("FRANKA_DESK_PASSWORD")
    if env_user and env_pass:
        return env_user, env_pass

    path = creds_path or _DEFAULT_CREDS_PATH
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        user = data.get("user")
        password = data.get("password")
        if not (user and password):
            logger.warning(
                "Desk credentials file %s missing 'user' or 'password'", path
            )
            return None
        return str(user), str(password)
    except Exception as exc:
        logger.warning("Could not read Desk credentials at %s: %s", path, exc)
        return None


def auto_unlock_and_enable_fci(
    fci_ip: str,
    creds_path: Optional[pathlib.Path] = None,
    force_take_control: bool = True,
) -> bool:
    """
    Open a Desk session, take control, unlock brakes, enable FCI.

    Returns True on full success, False otherwise (also logs the reason).
    Never raises: callers can call this unconditionally and proceed with
    the franky.Robot construction regardless. If brakes were already
    released and FCI already active, the calls are no-ops on the Control.
    """
    creds = _load_credentials(creds_path)
    if creds is None:
        logger.info(
            "Desk auto-unlock skipped: no credentials. Set "
            "FRANKA_DESK_USER + FRANKA_DESK_PASSWORD or create %s",
            _DEFAULT_CREDS_PATH,
        )
        return False
    user, password = creds

    if franky is None:
        logger.warning("Desk auto-unlock skipped: franky not importable")
        return False

    try:
        ws = franky.RobotWebSession(fci_ip, user, password)
    except Exception as exc:
        logger.warning("Desk auto-unlock skipped: session construct failed: %s", exc)
        return False

    try:
        ws.open()
    except Exception as exc:
        logger.warning("Desk auto-unlock: open() failed: %s", exc)
        return False

    try:
        try:
            has = getattr(ws, "has_control", None)
            already = (
                bool(has() if callable(has) else has) if has is not None else False
            )
            if already:
                logger.info("Desk auto-unlock: session already has control")
            else:
                # franky's default wait_timeout is 30 s, which stalls server
                # startup whenever another Desk session holds the lock. The
                # robot is usually already unlocked + FCI-active in that
                # case, so fail fast and let startup continue.
                ws.take_control(wait_timeout=5.0, force=force_take_control)
        except Exception as exc:
            logger.warning(
                "Desk auto-unlock: take_control failed (%s). Another "
                "Desk session may have the lock; try closing browser "
                "tabs pointed at %s or rerun with --auto-unlock-force.",
                exc,
                fci_ip,
            )
            return False

        try:
            ws.unlock_brakes()
            logger.info("Desk auto-unlock: brakes released")
        except Exception as exc:
            # Brakes already released is not an error from the Control's
            # standpoint but franky may surface it as one. Continue.
            logger.info(
                "Desk auto-unlock: unlock_brakes -> %s (likely already released)", exc
            )

        try:
            ws.enable_fci()
            logger.info("Desk auto-unlock: FCI enabled")
        except Exception as exc:
            logger.warning("Desk auto-unlock: enable_fci failed: %s", exc)
            return False

        return True
    finally:
        try:
            ws.close()
        except Exception:
            pass
