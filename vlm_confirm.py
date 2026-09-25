import base64
import json

import requests

import config
from logger import logger


def confirm(snapshot_path: str, danger_type: str) -> tuple:
    """Ritorna (confermato: bool, descrizione: str). In caso di errore, confermato=True (fail-open)."""
    if not config.VLM_ENABLED:
        return True, ""

    question = config.VLM_QUESTIONS.get(danger_type, config.VLM_DEFAULT_QUESTION)
    prompt = (
        f"{question} Rispondi SEMPRE e SOLO in lingua italiana, in JSON valido con questo formato esatto: "
        '{"pericolo_reale": true o false, "descrizione": "breve descrizione in italiano di cosa vedi, una frase"}'
    )

    try:
        with open(snapshot_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        resp = requests.post(
            config.VLM_ENDPOINT,
            json={
                "model": config.VLM_MODEL,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
            },
            timeout=config.VLM_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()

        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            logger.warning(f"VLM risposta senza JSON, fail-open: {raw[:200]}")
            return True, raw[:200]

        parsed = json.loads(raw[start:end + 1])
        confirmed = bool(parsed.get("pericolo_reale", True))
        description = str(parsed.get("descrizione", ""))
        logger.info(f"VLM [{danger_type}] confermato={confirmed}: {description}")
        return confirmed, description

    except Exception as e:
        logger.warning(f"VLM non disponibile/errore ({e}), evento passa senza conferma.")
        return True, ""
