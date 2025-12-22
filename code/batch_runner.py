
#!/usr/bin/env python3
"""
batch_runner.py

Run all problems in the ./data folder multiple times and collect results
for AMPL and Python into two CSV files.
"""

import os
import json
import csv
import argparse
import re
import shutil
from typing import List, Dict, Any, Tuple

from ampl_generator import generate_ampl_files, parse_ampl_solution
from python_generator import generate_gurobi_code
from data_transfer import data_transfer
from nl_to_structured import generate_prompt as struct_generate_prompt, process_prompt as struct_process_prompt


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def read_text(path: str, default: str = "") -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return default


def read_json(path: str, default: Any = None) -> Any:
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def list_problem_dirs(data_root: str) -> List[str]:
    dirs = []
    for name in os.listdir(data_root):
        full = os.path.join(data_root, name)
        if os.path.isdir(full):
            dirs.append(name)

    def _key(x: str):
        try:
            return int(x)
        except ValueError:
            return x

    return sorted(dirs, key=_key)


def classify_flags_ampl(sol: Dict[str, Any]) -> Tuple[int, int]:
    status = (sol.get("status") or "").lower()
    err_type = (sol.get("error_type") or "").lower()
    msg = (sol.get("error_message") or "").lower()
    obj_val = sol.get("objective_value")

    compile_flag = 0
    runtime_flag = 0

    if status == "syntax_error" or err_type == "compilation":
        compile_flag = 1
        return compile_flag, runtime_flag

    keywords = ["infeasible", "no feasible", "no solution", "unbounded", "failed"]
    if status in ("infeasible", "error", "unknown", "license_error"):
        runtime_flag = 1
    elif err_type == "runtime":
        runtime_flag = 1
    elif any(k in msg for k in keywords):
        runtime_flag = 1
    elif obj_val is None:
        runtime_flag = 1

    return compile_flag, runtime_flag


def classify_flags_python(sol: Dict[str, Any]) -> Tuple[int, int]:
    status = (sol.get("status") or "").lower()
    err_type = (sol.get("error_type") or "").lower()
    msg = (sol.get("error_message") or "").lower()
    obj_val = sol.get("objective_value")

    compile_flag = 0
    runtime_flag = 0

    if status == "syntax_error" or err_type == "compilation":
        compile_flag = 1
        return compile_flag, runtime_flag

    keywords = ["infeasible", "no feasible", "no solution", "unbounded", "failed"]
    if status in ("infeasible", "error", "unknown"):
        runtime_flag = 1
    elif err_type == "runtime":
        runtime_flag = 1
    elif any(k in msg for k in keywords):
        runtime_flag = 1
    elif obj_val is None:
        runtime_flag = 1

    return compile_flag, runtime_flag


def maybe_structure_problem(
    problem_dir: str,
    logs_root: str,
    model: str,
    use_logprobs: bool,
    do_structure: bool
) -> Dict[str, Any]:
    struct_path = os.path.join(problem_dir, "structured_description.json")

    if os.path.exists(struct_path):
        return read_json(struct_path, default={})

    if not do_structure:
        return {}

    description = read_text(os.path.join(problem_dir, "description.txt"), "")
    data_obj = read_json(os.path.join(problem_dir, "data.json"), default={})
    data_str = json.dumps(data_obj, indent=2)

    struct_log_dir = ensure_dir(os.path.join(logs_root, "struct", os.path.basename(problem_dir)))

    prompt = struct_generate_prompt(description, data_str)
    structured = struct_process_prompt(
        prompt,
        model=model,
        log_dir=struct_log_dir,
        use_logprobs=use_logprobs,
        run_number=1,
    )

    targets = {
        "description": structured.get("description", description),
        "objective": structured.get("objective", ""),
        "constraints": structured.get("constraints", []),
        "parameter": structured.get("parameter", []),
    }
    with open(struct_path, "w", encoding="utf-8") as f:
        json.dump(targets, f, indent=2)

    return targets


def prepare_run_input(
    problem_dir: str,
    run_root: str,
    structured_targets: Dict[str, Any]
) -> str:
    input_dir = ensure_dir(os.path.join(run_root, "input"))

    desc_text = read_text(os.path.join(problem_dir, "description.txt"), "")
    with open(os.path.join(input_dir, "description.txt"), "w", encoding="utf-8") as f:
        f.write(desc_text)

    data_obj = read_json(os.path.join(problem_dir, "data.json"), default={})
    with open(os.path.join(input_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump(data_obj, f, indent=2)

    ampl_txt_path = os.path.join(problem_dir, "ampl_data.txt")
    data_dat_path = os.path.join(problem_dir, "data.dat")

    ampl_text = ""
    if os.path.exists(ampl_txt_path):
        ampl_text = read_text(ampl_txt_path, "")
    elif os.path.exists(data_dat_path):
        ampl_text = read_text(data_dat_path, "")

    if ampl_text:
        with open(os.path.join(input_dir, "ampl_data.txt"), "w", encoding="utf-8") as f:
            f.write(ampl_text)
        with open(os.path.join(input_dir, "data.dat"), "w", encoding="utf-8") as f:
            f.write(ampl_text)
    else:
        with open(os.path.join(input_dir, "ampl_data.txt"), "w", encoding="utf-8") as f:
            f.write(
                "# Optional AMPL-style data context.\n"
                "# If empty, the model infers data usage from data.json.\n"
            )

    if structured_targets:
        with open(os.path.join(input_dir, "structured_description.json"), "w", encoding="utf-8") as f:
            json.dump(structured_targets, f, indent=2)
    else:
        struct_src = os.path.join(problem_dir, "structured_description.json")
        if os.path.exists(struct_src):
            with open(struct_src, "r", encoding="utf-8") as fsrc, \
                 open(os.path.join(input_dir, "structured_description.json"), "w", encoding="utf-8") as fdst:
                fdst.write(fsrc.read())

    return input_dir


def ensure_dat_copy(input_dir: str) -> None:
    dat_out = os.path.join(input_dir, "data.dat")
    ampl_txt = os.path.join(input_dir, "ampl_data.txt")

    dat_ok = os.path.exists(dat_out) and os.stat(dat_out).st_size > 0
    if dat_ok:
        return

    txt_ok = os.path.exists(ampl_txt) and os.stat(ampl_txt).st_size > 0
    if txt_ok:
        shutil.copy(ampl_txt, dat_out)


FIELDNAMES = [
    "#problem",
    "#run",
    "obj",
    "solution",
    "error_message",
    "Compile error",
    "Runtime Error",
    "llm_calls",
]


def append_csv_row(path: str, row: Dict[str, Any]) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_single_problem(
    problem_id: str,
    problem_dir: str,
    logs_root: str,
    n_runs: int,
    model: str,
    refinement: bool,
    max_refine: int,
    use_logprobs: bool,
    do_structure: bool,
    do_transfer: bool,
    ampl_csv_path: str,
    python_csv_path: str,
    run_ampl: bool = True,
    run_python: bool = True,
) -> Tuple[int, int]:
    ampl_count = 0
    python_count = 0

    obj_text = read_text(os.path.join(problem_dir, "obj.txt"), "").strip()

    structured_targets = maybe_structure_problem(
        problem_dir=problem_dir,
        logs_root=logs_root,
        model=model,
        use_logprobs=use_logprobs,
        do_structure=do_structure,
    )

    for run_idx in range(1, n_runs + 1):
        run_root = ensure_dir(os.path.join(logs_root, str(problem_id), str(run_idx)))
        input_dir = prepare_run_input(problem_dir, run_root, structured_targets)

        if do_transfer:
            data_transfer(input_dir, model_name=model, log_dir=run_root)
        else:
            ensure_dat_copy(input_dir)

        if run_ampl:
            ampl_log_dir = ensure_dir(os.path.join(run_root, "ampl"))
            _, _, final_solution_file, _, ampl_llm_calls = generate_ampl_files(
                input_dir=input_dir,
                model=model,
                log_dir=ampl_log_dir,
                use_logprobs=use_logprobs,
                run_number=run_idx,
                refinement=refinement,
                max_refine=max_refine,
            )

            ampl_status = parse_ampl_solution(final_solution_file)
            ampl_error_message = ampl_status.get("error_message", "")

            ampl_obj_val = ampl_status.get("objective_value")
            if ampl_obj_val in (None, "N/A"):
                content = read_text(final_solution_file, "")
                m = re.search(r"objective\s+([-+]?\d*\.?\d+)", content, re.IGNORECASE)
                if m:
                    try:
                        ampl_obj_val = float(m.group(1))
                    except Exception:
                        ampl_obj_val = m.group(1)

            ampl_solution_value = ""
            if ampl_obj_val not in (None, "N/A"):
                try:
                    ampl_solution_value = str(float(ampl_obj_val))
                except Exception:
                    ampl_solution_value = str(ampl_obj_val)

            ampl_compile_flag, ampl_runtime_flag = classify_flags_ampl(ampl_status)

            ampl_row = {
                "#problem": str(problem_id),
                "#run": run_idx,
                "obj": obj_text,
                "solution": ampl_solution_value,
                "error_message": ampl_error_message.replace("\n", "\\n"),
                "Compile error": ampl_compile_flag,
                "Runtime Error": ampl_runtime_flag,
                "llm_calls": ampl_llm_calls,
            }
            append_csv_row(ampl_csv_path, ampl_row)
            ampl_count += 1

        if run_python:
            python_log_dir = ensure_dir(os.path.join(run_root, "python"))
            _, solution_json_path, _, py_llm_calls = generate_gurobi_code(
                input_dir=input_dir,
                model=model,
                log_dir=python_log_dir,
                use_logprobs=use_logprobs,
                refinement=refinement,
                max_refine=max_refine,
                structure=False,
            )

            py_status = read_json(solution_json_path, default={})
            py_error_message = (py_status.get("error_message") or "")
            py_obj_val = py_status.get("objective_value")

            py_solution_value = ""
            if py_obj_val is not None:
                try:
                    py_solution_value = str(float(py_obj_val))
                except Exception:
                    py_solution_value = str(py_obj_val)

            py_compile_flag, py_runtime_flag = classify_flags_python(py_status)

            python_row = {
                "#problem": str(problem_id),
                "#run": run_idx,
                "obj": obj_text,
                "solution": py_solution_value,
                "error_message": py_error_message.replace("\n", "\\n"),
                "Compile error": py_compile_flag,
                "Runtime Error": py_runtime_flag,
                "llm_calls": py_llm_calls,
            }
            append_csv_row(python_csv_path, python_row)
            python_count += 1

    return ampl_count, python_count


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for problems under ./data, with AMPL and Python outputs."
    )
    parser.add_argument("--data_root", default="data", help="Root folder with problem subfolders (default: data)")
    parser.add_argument("--logs_root", default="logs", help="Root folder for logs and CSV outputs (default: logs)")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per problem (default: 5)")
    parser.add_argument("--model", default="gpt-4-1106-preview", help="LLM model name")
    parser.add_argument("--refinement", action="store_true", help="Use refinement loops in generators")
    parser.add_argument("--max_refine", type=int, default=1, help="Maximum number of refinement attempts (default: 1)")
    parser.add_argument("--logprob", action="store_true", help="Use log probability logging in LLM calls")
    parser.add_argument("--structure", action="store_true", help="Run NL structuring once per problem")
    parser.add_argument(
        "--no_transfer",
        action="store_true",
        help="Skip data transfer. Keep data.json, copy ampl_data.txt to data.dat.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--only_ampl", action="store_true", help="Run only the AMPL pipeline")
    group.add_argument("--only_python", action="store_true", help="Run only the Python pipeline")

    args = parser.parse_args()

    run_ampl = not args.only_python
    run_python = not args.only_ampl
    do_transfer = not args.no_transfer

    data_root = args.data_root
    logs_root = ensure_dir(args.logs_root)
    n_runs = args.runs

    ampl_csv_path = os.path.join(logs_root, "ampl_results.csv")
    python_csv_path = os.path.join(logs_root, "python_results.csv")

    if os.path.exists(ampl_csv_path):
        os.remove(ampl_csv_path)
    if os.path.exists(python_csv_path):
        os.remove(python_csv_path)

    problem_ids = list_problem_dirs(data_root)

    for pid in problem_ids:
        p_dir = os.path.join(data_root, pid)
        run_single_problem(
            problem_id=pid,
            problem_dir=p_dir,
            logs_root=logs_root,
            n_runs=n_runs,
            model=args.model,
            refinement=args.refinement,
            max_refine=args.max_refine,
            use_logprobs=args.logprob,
            do_structure=args.structure,
            do_transfer=do_transfer,
            ampl_csv_path=ampl_csv_path,
            python_csv_path=python_csv_path,
            run_ampl=run_ampl,
            run_python=run_python,
        )

    if run_ampl:
        print(f"AMPL results written to: {ampl_csv_path}")
    if run_python:
        print(f"Python results written to: {python_csv_path}")


if __name__ == "__main__":
    main()

