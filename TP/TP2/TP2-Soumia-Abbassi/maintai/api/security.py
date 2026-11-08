"""Authentification par clé API."""

import secrets

from fastapi import Header, HTTPException

from maintai.config import get_settings


def require_api_key(
    x_api_key: str | None = Header(default=None)
) -> None:
    """Vérifie la clé API envoyée par le client."""

    expected_key = get_settings().api_key

    # Vérifier si le serveur possède une clé configurée
    if expected_key is None:
        raise HTTPException(
            status_code=500,
            detail="Clé API non configurée sur le serveur"
        )

    # Vérifier si la clé est absente ou incorrecte
    if x_api_key is None or not secrets.compare_digest(
        x_api_key, expected_key
    ):
        raise HTTPException(
            status_code=401,
            detail="Clé API absente ou invalide"
        )