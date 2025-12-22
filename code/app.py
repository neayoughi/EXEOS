# app_backend.py
import os
import json
import time
import argparse
from typing import Optional, Dict, Any

from nl_to_structured import generate_prompt as struct_generate_prompt, process_prompt as struct_process_prompt
from ampl_generator import generate_ampl_files, parse_ampl_solution
from python_generator import generate_gurobi_code
from data_transfer import data_transfer

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text if isinstance(text, str) else json.dumps(text, indent=2))

def build_input_bundle(base_dir: str,
                       nl_description: str,
                       data_obj: Optional[Dict[str, Any]],
                       ampl_dat_text: Optional[str]) -> Dict[str, str]:
    """
    Create the input directory with all files expected by the generators.
    Accept JSON data and optional AMPL .dat text to seed AMPL prompts.
    Return paths for input_dir and log_dir.
    """
    ts = time.strftime('%Y%m%d_%H%M%S')
    log_dir = _ensure_dir(os.path.join("logs", f"run_{ts}"))
    input_dir = _ensure_dir(os.path.join(log_dir, "input"))

    # Primary inputs
    _write(os.path.join(input_dir, "description.txt"), nl_description)
    _write(os.path.join(input_dir, "data.json"), data_obj or {})

    # Optional AMPL .dat seed
    if ampl_dat_text and ampl_dat_text.strip():
        _write(os.path.join(input_dir, "ampl_data.txt"), ampl_dat_text)
        _write(os.path.join(input_dir, "data.dat"), ampl_dat_text)
    else:
        _write(
            os.path.join(input_dir, "ampl_data.txt"),
            "# Optional AMPL-style data context for the LLM.\n"
            "# If empty, the model infers data usage from data.json.\n"
        )
    return {"log_dir": log_dir, "input_dir": input_dir}

def run_pipeline(nl_description: str,
                 data: Optional[Dict[str, Any]] = None,
                 ampl_dat_text: Optional[str] = None,
                 model: str = "gpt-4-1106-preview",
                 refinement: bool = True,
                 max_refine: int = 2,
                 structure: bool = False,
                 use_logprobs: bool = False) -> Dict[str, Any]:
    """
    Orchestrate: NL → optional structuring → AMPL + Python generation → solve → refinement → results.
    """
    bundles = build_input_bundle(base_dir=".",
                                 nl_description=nl_description,
                                 data_obj=data,
                                 ampl_dat_text=ampl_dat_text)
    input_dir, log_dir = bundles["input_dir"], bundles["log_dir"]

    structured: Dict[str, Any] = {}
    s_path_input = os.path.join(input_dir, "structured_description.json")

    if structure:
        prompt = struct_generate_prompt(nl_description, json.dumps(data or {}, indent=2))
        structured = struct_process_prompt(
            prompt,
            model=model,
            log_dir=log_dir,
            use_logprobs=use_logprobs,
            run_number=1
        )
        targets = {
            "description": structured.get("description", nl_description),
            "objective": structured.get("objective", ""),
            "constraints": structured.get("constraints", []),
            "parameter": structured.get("parameter", [])
        }
        _write(s_path_input, targets)
    else:
        # Plain fallback, keep the same schema
        targets = {
            "description": nl_description,
            "objective": "",
            "constraints": [],
            "parameter": []
        }
        _write(s_path_input, targets)

    data_transfer(input_dir, model_name=model, log_dir=log_dir)

    # AMPL generation → solve → refinement
    ampl_mod, ampl_dat, ampl_solution_file, ampl_refine_attempts, ampl_llm_calls = generate_ampl_files(
        input_dir=input_dir,
        model=model,
        log_dir=_ensure_dir(os.path.join(log_dir, "ampl")),
        use_logprobs=use_logprobs,
        run_number=1,
        refinement=refinement,
        max_refine=max_refine
    )
    ampl_status = parse_ampl_solution(ampl_solution_file)

    # Python (Gurobi) generation → solve → refinement
    py_code_path, py_solution_json, py_attempts, py_llm_calls = generate_gurobi_code(
        input_dir=input_dir,
        model=model,
        log_dir=_ensure_dir(os.path.join(log_dir, "python")),
        use_logprobs=use_logprobs,
        refinement=refinement,
        max_refine=max_refine
    )
    with open(py_solution_json, "r", encoding="utf-8") as f:
        py_status = json.load(f)

    # Package results
    result = {
        "log_dir": log_dir,
        "structured": structured,
        "ampl": {
            "model_path": ampl_mod,
            "data_path": ampl_dat,
            "solution_path": ampl_solution_file,
            "status": ampl_status,
            "refinement_attempts": ampl_refine_attempts,
            "llm_calls": ampl_llm_calls
        },
        "python": {
            "code_path": py_code_path,
            "solution_path": py_solution_json,
            "status": py_status,
            "refinement_attempts": py_attempts,
            "llm_calls": py_llm_calls
        }
    }
    with open(os.path.join(log_dir, "pipeline_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result

def main():
    parser = argparse.ArgumentParser(description="End-to-end NL → Structured → AMPL+Python → Solve pipeline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--nl_file", help="Path to a .txt file with the NL optimization description")
    group.add_argument("--nl", help="NL optimization description as a raw string")

    # Accept one or two files after --data_file: JSON first, optional AMPL .dat second
    parser.add_argument(
        "--data_file",
        nargs="+",
        help="Provide data.json and optionally data.dat. Example: --data_file data.json data.dat"
    )

    parser.add_argument("--model", default="gpt-4-1106-preview")
    parser.add_argument("--structure", action="store_true")
    parser.add_argument("--refinement", action="store_true")
    parser.add_argument("--max_refine", type=int, default=2)
    parser.add_argument("--logprob", action="store_true", help="Enable log probability logging across the pipeline")
    args = parser.parse_args()

    nl_text = open(args.nl_file, "r", encoding="utf-8").read() if args.nl_file else args.nl

    data_obj: Dict[str, Any] = {}
    ampl_dat_text: Optional[str] = None
    if args.data_file:
        for path in args.data_file:
            if path.lower().endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    data_obj = json.load(f)
            # elif path.lower().endswith(".dat"):
            elif path.lower().endswith((".dat", ".txt")):
                with open(path, "r", encoding="utf-8") as f:
                    ampl_dat_text = f.read()

    out = run_pipeline(
        nl_text,
        data=data_obj,
        ampl_dat_text=ampl_dat_text,
        model=args.model,
        refinement=args.refinement,
        max_refine=args.max_refine,
        structure=args.structure,
        use_logprobs=args.logprob
    )
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
