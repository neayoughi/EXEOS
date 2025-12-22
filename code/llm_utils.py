import json
import os
import time
from typing import Optional, Sequence

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_google_vertexai import ChatVertexAI
import google.auth
from vertexai.preview.generative_models import GenerativeModel
from google.protobuf.json_format import MessageToDict

# Optional google-genai imports for Gemini 3 thinking control
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # optional dependency
    genai = None
    genai_types = None

# Load config once
_CONFIG_PATH = os.getenv("EXEOS_CONFIG_PATH", "config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)


def _cfg(key: str, default: Optional[str] = None) -> Optional[str]:
    v = CONFIG.get(key)
    return v if (v is not None and str(v) != "") else default


def _vertex_model_id_for_alias(model_id: str) -> str:
    """
    Map friendly aliases to actual Vertex model ids.
    """
    # CLI uses: --model vertex-gemini-3-pro
    # Actual Vertex model id is:
    #   gemini-3-pro-preview
    if model_id == "gemini-3-pro":
        return "gemini-3-pro-preview"
    return model_id


def _vertex_location_for_model(model_id: str) -> str:
    """
    Decide which Vertex location to use for a given model.

    Default comes from CONFIG["gcp_location"], but some models
    such as Gemini 3 use the global endpoint.
    """
    base_loc = _cfg("gcp_location", "us-central1")

    # Models that use the global endpoint
    global_models = {
        "gemini-3-pro-preview",
        "gemini-3-pro",           # safety, in case alias is used directly
        "gemini-3-pro-image-preview",
    }

    if model_id in global_models:
        return "global"

    return base_loc


def _is_gemini3(model_id: str) -> bool:
    return model_id.startswith("gemini-3")


def _gemini3_thinking_level() -> str:
    """
    Read desired Gemini 3 thinking level from config.

    Supported values in config:
      - "LOW"  (or anything else) → low thinking
      - "HIGH"                     → high thinking
    """
    raw = _cfg("gemini3_thinking_level", "LOW")
    if raw is None:
        return "LOW"

    s = str(raw).strip().upper()
    return "HIGH" if s == "HIGH" else "LOW"


def get_llm(model: str):
    """Return a ChatOpenAI or ChatVertexAI instance based on model prefix."""
    if model.startswith("gpt") or model == "o4-mini":
        return ChatOpenAI(
            model_name=model,
            openai_api_key=_cfg("openai_api_key"),
            organization=_cfg("openai_org_id"),
        )

    if model.startswith("vertex-"):
        # Example: "vertex-gemini-3-pro" → "gemini-3-pro-preview"
        raw_id = model.replace("vertex-", "", 1).strip()
        model_id = _vertex_model_id_for_alias(raw_id)
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        location = _vertex_location_for_model(model_id)
        return ChatVertexAI(
            model_name=model_id,
            project=_cfg("gcp_project_id"),
            location=location,
            credentials=credentials,
        )

    raise ValueError(f"Unsupported model: {model}")


def _gemini3_call_via_genai(
    model_name: str,
    messages: Sequence[BaseMessage],
    use_logprobs: bool,
    log_dir: Optional[str],
) -> Optional[str]:
    """
    Handle Gemini 3 calls through the google-genai client.

    This path provides explicit thinking control.
    Logprobs are not available for Gemini 3, so the use_logprobs flag
    controls only whether basic usage metadata is written to disk.
    """
    if genai is None or genai_types is None:
        print(
            "[DEBUG] google-genai SDK not available; "
            "falling back to ChatVertexAI.invoke without explicit thinking control."
        )
        return None  # Caller must handle fallback

    project_id = _cfg("gcp_project_id")
    location = _vertex_location_for_model(model_name)

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )

    # Flatten LangChain messages into a single text prompt
    text_prompt = "\n".join(
        [m.content for m in messages if getattr(m, "content", "")]
    )

    # Configure thinking level
    lvl = _gemini3_thinking_level()
    if lvl == "HIGH":
        level_enum = genai_types.ThinkingLevel.HIGH
    else:
        level_enum = genai_types.ThinkingLevel.LOW

    config = genai_types.GenerateContentConfig(
        thinking_config=genai_types.ThinkingConfig(
            thinking_level=level_enum
        )
    )

    response = client.models.generate_content(
        model=model_name,
        contents=text_prompt,
        config=config,
    )

    content = response.text or ""

    # Optionally persist usage metadata when "logprobs" mode is requested
    if use_logprobs and log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            usage = getattr(response, "usage_metadata", None)
            meta = {
                "full_text": content,
                "model": model_name,
                "thinking_level": lvl,
            }
            if usage is not None:
                meta.update(
                    {
                        "prompt_token_count": usage.prompt_token_count,
                        "candidates_token_count": usage.candidates_token_count,
                        "thoughts_token_count": getattr(
                            usage, "thoughts_token_count", None
                        ),
                        "total_token_count": usage.total_token_count,
                    }
                )
            out_path = os.path.join(log_dir, f"gemini3_usage_{ts}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(f"[DEBUG] Gemini 3 usage metadata written to {out_path}")
        except Exception as e:
            print("[DEBUG] Failed to write Gemini 3 usage metadata:", e)

    return content


def llm_call(
    llm,
    messages: Sequence[BaseMessage],
    use_logprobs: bool = False,
    log_dir: Optional[str] = None,
    top_logprobs: int = 5,
) -> str:
    """
    Wrapper for both OpenAI and Gemini (Vertex) LLMs.
    When use_logprobs=True and supported, saves token-level probabilities.
    """

    # Gemini 3 handled here through google-genai, with thinking control.
    if isinstance(llm, ChatVertexAI):
        model_name = getattr(llm, "model_name", "")
        if _is_gemini3(model_name):
            content = _gemini3_call_via_genai(
                model_name=model_name,
                messages=messages,
                use_logprobs=use_logprobs,
                log_dir=log_dir,
            )
            if content is not None:
                return content
            # If google-genai is unavailable, fall through to default paths.

    # ---- Gemini / Vertex AI direct call for logprobs (non-Gemini 3) ----
    if use_logprobs and isinstance(llm, ChatVertexAI):
        try:
            model_name = llm.model_name
            gen_model = GenerativeModel(model_name)
            text_prompt = "\n".join(
                [m.content for m in messages if getattr(m, "content", "")]
            )

            # Ask Gemini for top-k logprobs
            response = gen_model.generate_content(
                [text_prompt],
                generation_config={
                    "response_logprobs": True,
                    "logprobs": top_logprobs,
                },
                safety_settings=[],
            )

            content = getattr(response, "text", "") or ""
            lp_data = {}

            # Extract structured logprobs
            if response.candidates and getattr(
                response.candidates[0], "logprobs_result", None
            ):
                lp_obj = response.candidates[0].logprobs_result
                lp_dict = MessageToDict(lp_obj._pb)

                print(
                    "[DEBUG] Gemini logprobs_result captured with "
                    f"{len(lp_dict.get('topCandidates', []))} token positions"
                )

                lp_data = {
                    "full_text": content or response.candidates[0].text,
                    "logprobs_result": lp_dict,
                }

                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    out_path = os.path.join(
                        log_dir, f"gemini_logprobs_{ts}.json"
                    )
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(lp_data, f, indent=2)
                    print(
                        f"[DEBUG] Gemini logprobs JSON written to {out_path}"
                    )
            else:
                print("[DEBUG] Gemini response had no logprobs_result field")

            return content or (
                response.candidates[0].text if response.candidates else ""
            )

        except Exception as e:
            print("[DEBUG] Gemini call failed:", e)
            resp = llm.invoke(messages)
            return resp.content

    # ---- Default path: OpenAI / generic chat models ----
    resp = llm.invoke(messages)
    content = resp.content

    # ---- OpenAI logprobs ----
    if use_logprobs and hasattr(llm, "generate"):
        try:
            result = llm.generate(
                [list(messages)],
                logprobs=True,
                top_logprobs=top_logprobs,
            )
            gen = result.generations[0][0]
            lp = (gen.generation_info or {}).get("logprobs")
            if lp and log_dir:
                os.makedirs(log_dir, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                out = {
                    "full_text": gen.message.content,
                    "tokens": {"content": lp.get("content", [])},
                }
                out_path = os.path.join(log_dir, f"openai_logprobs_{ts}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2)
                print(f"[DEBUG] OpenAI logprobs written to {out_path}")
        except Exception as e:
            print("[DEBUG] OpenAI logprobs failed:", e)

    return content

