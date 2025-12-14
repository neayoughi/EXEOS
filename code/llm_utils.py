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

# Load config once
_CONFIG_PATH = os.getenv("EXEOS_CONFIG_PATH", "config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

def _cfg(key: str, default: Optional[str] = None) -> Optional[str]:
    v = CONFIG.get(key)
    return v if (v is not None and str(v) != "") else default


def get_llm(model: str):
    """Return a ChatOpenAI or ChatVertexAI instance based on model prefix."""
    if model.startswith("gpt") or model == "o4-mini":
        return ChatOpenAI(
            model_name=model,
            openai_api_key=_cfg("openai_api_key"),
            organization=_cfg("openai_org_id")
        )

    if model.startswith("vertex-"):
        model_id = model.replace("vertex-", "", 1).strip()
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return ChatVertexAI(
            model_name=model_id,
            project=_cfg("gcp_project_id"),
            location=_cfg("gcp_location", "us-central1"),
            credentials=credentials
        )

    raise ValueError(f"Unsupported model: {model}")

def llm_call(
    llm,
    messages: Sequence[BaseMessage],
    use_logprobs: bool = False,
    log_dir: Optional[str] = None,
    top_logprobs: int = 5
) -> str:
    """
    Wrapper for both OpenAI and Gemini (Vertex) LLMs.
    When use_logprobs=True and supported, saves token-level probabilities.
    """

    # ---- Gemini / Vertex AI direct call ----
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
            if response.candidates and getattr(response.candidates[0], "logprobs_result", None):
                lp_obj = response.candidates[0].logprobs_result
                lp_dict = MessageToDict(lp_obj._pb)

                print(f"[DEBUG] Gemini logprobs_result captured with "
                      f"{len(lp_dict.get('topCandidates', []))} token positions")

                lp_data = {
                    "full_text": content or response.candidates[0].text,
                    "logprobs_result": lp_dict,
                }

                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    out_path = os.path.join(log_dir, f"gemini_logprobs_{ts}.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(lp_data, f, indent=2)
                    print(f"[DEBUG] Gemini logprobs JSON written to {out_path}")
            else:
                print("[DEBUG] Gemini response had no logprobs_result field")

            return content or (response.candidates[0].text if response.candidates else "")

        except Exception as e:
            print("[DEBUG] Gemini call failed:", e)
            resp = llm.invoke(messages)
            return resp.content

    # ---- Default path: OpenAI / no logprobs ----
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


    # ---- Default path: OpenAI / no logprobs ----
    resp = llm.invoke(messages)
    content = resp.content

    # ---- OpenAI logprobs ----
    if use_logprobs and hasattr(llm, "generate"):
        try:
            result = llm.generate(
                [list(messages)],
                logprobs=True,
                top_logprobs=top_logprobs
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
                with open(
                    os.path.join(log_dir, f"logprobs_{ts}.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(out, f, indent=2)
        except Exception as e:
            print("[DEBUG] OpenAI logprobs failed:", e)

    return content

