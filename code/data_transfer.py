
# data_transfer.py  [LLM-only version]
import json
import os
import re
import shutil
from typing import Any, Dict, Optional, Tuple

from llm_utils import get_llm, llm_call
from langchain_core.messages import HumanMessage

# =========================
# LLM prompt
# =========================
# _PROMPT_TEMPLATE = """
# You will match user-provided data parameter names to symbols used in a target schema of parameters.

# ### TARGET STRUCTURE (read-only)
# {structured}

# ### USER DATA JSON (read-only; keys and shapes may not match target)
# {data_json}

# ### AMPL DATA TEMPLATE (read-only)
# {ampl_text}

# TASK

# 1) Produce a JSON object where:
#    - The top-level keys are exactly the parameter symbols from TARGET STRUCTURE.
#    - The values come from USER DATA JSON.
#    - You keep every numeric value, index shape, and ordering from USER DATA JSON.
#    - You only move data between keys when the meaning matches a symbol from TARGET STRUCTURE.

# 2) Produce an AMPL data text for the `.dat` file where:
#    - You start from the AMPL DATA TEMPLATE.
#    - You keep all numeric values, all indices, all dates, all line breaks, and the order of blocks exactly as in the template.
#    - You only change parameter and set names so they match the symbols and set names from TARGET STRUCTURE where that mapping is clear.
#    - You must not introduce new parameters, remove parameters, or change any numeric literal.
#    - You must not re-index or compress tables; every line after each parameter name must stay identical except for the renamed identifier at the beginning.

# 3) Parameters or sets that do not appear in TARGET STRUCTURE stay unchanged in both JSON and AMPL.

# OUTPUT FORMAT (strict):
# ---JSON---
# {{json here}}
# ---DAT---
# {{dat here}}
# """
_PROMPT_TEMPLATE = """
You will match user-provided data parameter names to symbols used in a target schema of parameters.

### TARGET STRUCTURE (read-only)
{structured}

### USER DATA JSON (read-only; keys and shapes may not match target)
{data_json}

### AMPL DATA TEMPLATE (read-only)
{ampl_text}

TASK

1) Produce a JSON object where:
   - The top-level keys are exactly the parameter symbols from TARGET STRUCTURE for which you can find data in USER DATA JSON.
   - The values come from USER DATA JSON.
   - You keep every numeric value, index shape, and ordering from USER DATA JSON. Do not change signs, scales, or magnitudes.
   - You only move data between keys when the meaning matches a symbol from TARGET STRUCTURE.

2) Produce an AMPL data text for the `.dat` file where:
   - You start from the AMPL DATA TEMPLATE.
   - You keep all numeric values, all indices, all dates, all line breaks, and the order of blocks exactly as in the template.
   - You only change parameter and set names so they match the symbols and set names from TARGET STRUCTURE where that mapping is clear.
   - If a parameter declaration in the template has no corresponding symbol in TARGET STRUCTURE, you remove that entire parameter block from the AMPL output.
   - You keep set declarations even when the set name does not appear in TARGET STRUCTURE.
   - You must not introduce new parameters, remove parameters that do correspond to TARGET STRUCTURE, or change any numeric literal.
   - You must not re-index or compress tables; every line after each parameter name must stay identical except for the renamed identifier at the beginning.

3) Parameters that do not appear in TARGET STRUCTURE must be omitted from the AMPL output.
   Sets that do not appear in TARGET STRUCTURE stay unchanged in AMPL.

OUTPUT FORMAT (strict):
---JSON---
{{json here}}
---DAT---
{{dat here}}
"""


# =========================
# IO helpers
# =========================
def _safe_read(path: str, default: str = "") -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return default

def _safe_read_json(path: str, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

# =========================
# Cleaner
# =========================
def _strip_code_fences(s: str) -> str:  # [NEW]
    s = s.strip()
    # remove common fences like ```json ... ```
    s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _extract_json_object(s: str) -> Optional[str]:  # [NEW]
    """Return the first valid top-level JSON object substring, or None."""
    start = s.find("{")
    if start == -1:
        return None
    # scan with a simple stack for braces
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start : i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    # keep scanning for another balanced block
                    pass
    return None

# def _clean_dat(s: str) -> str:  # [NEW]
#     """
#     Keep only AMPL-like param blocks.
#     Rules:
#       - Drop code fences, markdown, prose.
#       - Keep from first 'param ' onward.
#       - Remove leading non-param lines.
#       - Remove trailing non-; junk.
#     """
#     s = _strip_code_fences(s)
#     # If sections exist, we already isolated; else find from first 'param '
#     idx = s.lower().find("param ")
#     if idx >= 0:
#         s = s[idx:]
#     # Remove obvious markdown bullets and headings
#     lines = [ln for ln in s.splitlines() if not ln.strip().startswith(("#", "-", "*"))]
#     # Keep lines that look like AMPL param content or blank separators
#     kept = []
#     for ln in lines:
#         t = ln.strip()
#         if not t:
#             kept.append("")
#             continue
#         if t.lower().startswith("param ") or re.match(r"^\d+(\s+\S+)*;?$", t) or t.endswith(";") or ":" in t:
#             kept.append(ln)
#         # else drop
#     cleaned = "\n".join(kept).strip()
#     # Ensure it ends after the last semicolon if any
#     last_semi = cleaned.rfind(";")
#     if last_semi != -1:
#         cleaned = cleaned[: last_semi + 1]
#     return cleaned.strip()

def _clean_dat(s: str) -> str:
    """
    Keep AMPL-style set/param blocks.
    Rules:
      - Drop code fences, markdown, prose.
      - Keep from the first 'set ' or 'param ' onward.
      - Keep lines that look like AMPL set/param content.
    """
    s = _strip_code_fences(s)

    # Start from earliest occurrence of 'set ' or 'param '
    lower = s.lower()
    idx_set = lower.find("set ")
    idx_param = lower.find("param ")
    starts = [i for i in (idx_set, idx_param) if i >= 0]
    if starts:
        s = s[min(starts):]

    # Remove obvious markdown bullets and headings
    lines = [ln for ln in s.splitlines()
             if not ln.strip().startswith(("#", "-", "*"))]

    kept = []
    for ln in lines:
        t = ln.strip()
        if not t:
            kept.append("")
            continue

        # Keep set and param declarations
        if t.lower().startswith("set ") or t.lower().startswith("param "):
            kept.append(ln)
            continue

        # Keep rows that look like AMPL data or headers
        if re.match(r"^\S+(\s+\S+)*;?$", t) or ":" in t or t.endswith(";"):
            kept.append(ln)

    cleaned = "\n".join(kept).strip()

    # Truncate after last semicolon, if any
    last_semi = cleaned.rfind(";")
    if last_semi != -1:
        cleaned = cleaned[: last_semi + 1]

    return cleaned.strip()


def _split_sections(text: str) -> Tuple[str, str]:  # [NEW]
    """
    Return (json_str, dat_str) with aggressive sanitation.
    Priority: explicit markers. Fallback: best-effort extraction.
    """
    text = text.strip()
    # Fast path: explicit markers
    if "---JSON---" in text and "---DAT---" in text:
        a = text.split("---JSON---", 1)[1]
        json_part, dat_part = a.split("---DAT---", 1)
        json_part = _strip_code_fences(json_part).strip()
        dat_part = _strip_code_fences(dat_part).strip()
        # Validate JSON; if invalid, try to salvage
        try:
            json.loads(json_part)
        except Exception:
            salvaged = _extract_json_object(json_part) or ""
            json_part = salvaged
        dat_part = _clean_dat(dat_part)
        return json_part, dat_part

    # Fallback: salvage from raw
    json_part = _extract_json_object(text) or ""
    # For DAT, take content after JSON and clean
    dat_raw = text
    if json_part:
        end = text.find(json_part) + len(json_part)
        dat_raw = text[end:]
    dat_part = _clean_dat(dat_raw)
    return json_part, dat_part

# =========================
# Main (LLM-only)
# =========================
def data_transfer(input_dir: str, model_name: str = "gpt-4-1106-preview", log_dir: Optional[str] = None):
    """
    LLM-only normalization.
    Emit:
      - <input_dir>/data.json
      - <input_dir>/data.dat
      - <input_dir>/data_transfer_report.json
    """
    structured_path = os.path.join(input_dir, "structured_description.json")
    data_path       = os.path.join(input_dir, "data.json")
    ampl_txt_path   = os.path.join(input_dir, "ampl_data.txt")
    dat_out_path    = os.path.join(input_dir, "data.dat")
    report_path     = os.path.join(input_dir, "data_transfer_report.json")

    structured = _safe_read_json(structured_path, {"parameter": []})
    user_data  = _safe_read_json(data_path, {})
    ampl_text  = _safe_read(ampl_txt_path, "")

    # Build prompt
    prompt = _PROMPT_TEMPLATE.format(
        structured=json.dumps(structured, indent=2),
        data_json=json.dumps(user_data, indent=2),
        ampl_text=ampl_text[:6000]
    )

    # Call LLM
    llm = get_llm(model_name)
    llm_raw = llm_call(llm, [HumanMessage(content=prompt)], use_logprobs=False, log_dir=log_dir)

    # Clean and extract
    json_part, dat_part = _split_sections(llm_raw)
    parsed_json: Dict[str, Any] = {}

    # Parse JSON if possible
    json_ok = False
    if json_part:
        try:
            parsed_json = json.loads(json_part)
            json_ok = True
        except Exception:
            json_ok = False

    # Write outputs
    if json_ok:
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(parsed_json, f, indent=2)
    # If JSON failed, keep original data.json as-is

    dat_written = False
    if dat_part:
        with open(dat_out_path, "w", encoding="utf-8") as f:
            f.write(dat_part)
        dat_written = True
    elif os.path.exists(ampl_txt_path) and os.stat(ampl_txt_path).st_size > 0:
        shutil.copy(ampl_txt_path, dat_out_path)
        dat_written = True

    # Report
    report = {
        "llm_used": True,
        "json_extracted": json_ok,
        "json_keys": sorted(list(parsed_json.keys())) if json_ok else [],
        "dat_nonempty": dat_written,
        "llm_raw_preview": llm_raw[:1000]
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


