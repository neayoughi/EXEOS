import json
import os
import time
from typing import Optional, Sequence

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_vertexai import ChatVertexAI
import google.auth

# Load config once
_CONFIG_PATH = os.getenv("EXEOS_CONFIG_PATH", "config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

def _cfg(key: str, default: Optional[str] = None) -> Optional[str]:
    v = CONFIG.get(key)
    return v if (v is not None and str(v) != "") else default

def get_llm(model: str):
    """
    Single entry point. Picks provider from model name.
    Reads keys, project, and location from config.json.
    - OpenAI:   model startswith 'gpt' or equals 'o4-mini'
    - VertexAI: model startswith 'vertex-' (use the part after 'vertex-' as Vertex model name)
    """
    # OpenAI
    if model.startswith("gpt") or model == "o4-mini":
        return ChatOpenAI(
            model_name=model,
            openai_api_key=_cfg("openai_api_key"),
            # Pass org only if present in config to avoid SDK warnings
            organization=_cfg("openai_org_id")
        )

    # Vertex AI
    if model.startswith("vertex-"):
        model_id = model.replace("vertex-", "", 1).strip()
        # Credentials come from ADC (GOOGLE_APPLICATION_CREDENTIALS or local gcloud)
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        return ChatVertexAI(
            model_name=model_id,
            project=_cfg("gcp_project_id"),
            location=_cfg("gcp_location", "us-central1"),
            credentials=credentials
        )

    raise ValueError(f"Unsupported model: {model}")

def llm_call(llm, messages: Sequence[BaseMessage], use_logprobs: bool = False, log_dir: Optional[str] = None) -> str:
    """
    Simple wrapper. Uses .invoke for both providers.
    If use_logprobs is True and provider supports it, logs top tokens to a JSON file.
    """
    # Minimal path: call and return
    resp = llm.invoke(messages)
    content = resp.content

    # Best-effort logprob capture for OpenAI when available (non-fatal if unsupported)
    if use_logprobs and hasattr(llm, "generate"):
        try:
            result = llm.generate([list(messages)], logprobs=True, top_logprobs=5)
            gen = result.generations[0][0]
            lp = (gen.generation_info or {}).get("logprobs")
            if lp and log_dir:
                os.makedirs(log_dir, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                out = {
                    "full_text": gen.message.content,
                    "tokens": {"content": lp.get("content", [])}
                }
                with open(os.path.join(log_dir, f"logprobs_{ts}.json"), "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2)
        except Exception:
            # Ignore logprob failures
            pass

    return content
