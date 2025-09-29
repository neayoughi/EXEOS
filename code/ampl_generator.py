# ampl_generator.py
import argparse
import os
import json
import subprocess
import textwrap
import time
import re
from llm_utils import get_llm
from langchain_core.messages import HumanMessage, AIMessage
from llm_utils import llm_call

PROMPT_DIR = os.getenv("EXEOS_PROMPT_DIR", "prompt")

def _read_prompt(filename, default=""):
    p1 = os.path.join(PROMPT_DIR, filename)
    if os.path.exists(p1):
        return _safe_read(p1, default)
    return _safe_read(filename, default)

def _safe_read(path, default=""):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return default

def _load_targets(input_dir):
    """Load structured_description.json if present, else fall back to raw description."""
    p = os.path.join(input_dir, "structured_description.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    desc_text = _safe_read(os.path.join(input_dir, "description.txt"), "")
    return {"description": desc_text, "objective": "", "constraints": [], "parameter": []}

def _select_data_for_solve(input_dir: str, log_dir: str) -> str:
    """
    Choose the .dat used for solving. Prefer input/data.dat, else input/ampl_data.txt.
    Copy the chosen text into logs/.../ampl/data.dat and return that path.
    """
    dst = os.path.join(log_dir, "data.dat")
    p_dat = os.path.join(input_dir, "data.dat")
    p_txt = os.path.join(input_dir, "ampl_data.txt")
    if os.path.exists(p_dat):
        with open(p_dat, "r", encoding="utf-8") as fsrc, open(dst, "w", encoding="utf-8") as fdst:
            fdst.write(fsrc.read())
        return dst
    if os.path.exists(p_txt):
        with open(p_txt, "r", encoding="utf-8") as fsrc, open(dst, "w", encoding="utf-8") as fdst:
            fdst.write(fsrc.read())
        return dst
    # No user data present; create empty .dat so model can compile
    open(dst, "w", encoding="utf-8").close()
    return dst

def clean_code(code, is_model=False):
    """
    Clean AMPL code by removing unnecessary whitespace and separator markers.
    For model files, ensure ---MODEL--- and ---DATA--- markers are excluded.
    """
    code = code.strip()
    if is_model:
        if "---MODEL---" in code:
            code = code.split("---MODEL---")[0].strip()
        if "---DATA---" in code:
            code = code.split("---DATA---")[0].strip()
    code = re.sub(r'\s*;\s*$', '', code.strip())
    code = '\n'.join(line.strip() for line in code.splitlines() if line.strip())
    return code

def parse_ampl_solution(solution_path):
    """
    Parse AMPL solution file to determine status and extract relevant information.
    """
    try:
        with open(solution_path, 'r', encoding="utf-8") as f:
            content = f.read().lower()
    except FileNotFoundError:
        return {"status": "error", "error_message": f"Solution file {solution_path} not found.", "error_type": "Runtime", "runtime": 1, "compile": 0}
    if "optimal solution" in content:
        objective_match = re.search(r"objective\s+([\d\.]+)", content, re.IGNORECASE)
        objective_value = float(objective_match.group(1)) if objective_match else "N/A"
        return {"status": "optimal", "objective_value": objective_value, "error_message": "", "error_type": "None", "runtime": 0, "compile": 0}
    elif "license" in content and "expired" in content:
        return {"status": "license_error", "error_message": "AMPL license has expired. Please renew the license.", "error_type": "Runtime", "runtime": 1, "compile": 0}
    elif "syntax error" in content or "expected number" in content or "no variables declared" in content or "not a param" in content:
        return {"status": "syntax_error", "error_message": content, "error_type": "Compilation", "runtime": 0, "compile": 1}
    elif "error" in content or "exception" in content:
        return {"status": "error", "error_message": content, "error_type": "Runtime", "runtime": 1, "compile": 0}
    elif "gurobi" in content or "solver" in content or "presolve" in content or "infeasible" in content or " no feasible" in content:
        return {"status": "error", "error_message": content, "error_type": "Runtime", "runtime": 1, "compile": 0}
    else:
        return {"status": "unknown", "error_message": content, "error_type": "Runtime", "runtime": 0, "compile": 0}

def solve_ampl(log_dir, model_file='model.mod', data_file='data.dat', solution_file='ampl_solution.txt'):
    ampl_exec = os.path.join(log_dir, 'execute_ampl.py')
    ampl_code = textwrap.dedent(f"""\
    #!/usr/bin/env python
    # -*- coding: utf-8 -*-
    from amplpy import AMPL
    ampl = AMPL()
    ampl.reset()
    ampl.set_option('reset_initial_guesses', True)
    ampl.set_option('send_statuses', False)
    ampl.read('{model_file}')
    ampl.read_data('{data_file}')
    ampl.set_option('solver', 'gurobi')
    try:
        ampl.solve()
        variables = ampl.getVariables()
        result_str = ""
        for var_tuple in variables:
            var_name = var_tuple[0]
            var = variables[var_name]
            result_str += f"Variable: {{var_name}}\\n"
            try:
                values = var.getValues()
                result_str += values.toString() + "\\n\\n"
            except Exception as _:
                try:
                    result_str += str(var.value()) + "\\n\\n"
                except Exception as __:
                    pass
        print("optimal solution")
        print(result_str)
    except Exception as e:
        import sys
        print("Error:", str(e), file=sys.stderr)
    """)
    with open(ampl_exec, 'w', encoding="utf-8") as f:
        f.write(ampl_code)

    result_file_path = os.path.join(log_dir, solution_file)
    try:
        ar = subprocess.run(
            ['python3', 'execute_ampl.py'],
            capture_output=True, text=True,
            timeout=30, cwd=log_dir
        )
        raw_ampl = ar.stdout if ar.returncode == 0 else f"Error: {ar.stderr}"
    except subprocess.TimeoutExpired:
        raw_ampl = "Error: AMPL script execution timed out."
    except Exception as e:
        raw_ampl = f"AMPL execution failed: {e}"

    with open(result_file_path, "w", encoding="utf-8") as result_file:
        result_file.write(raw_ampl)

    return result_file_path

def generate_ampl_files(input_dir, model, log_dir, use_logprobs=False, run_number=None, refinement=False, max_refine=1):
    # Load targets prepared by backend
    target = _load_targets(input_dir)

    # Always use user-supplied data for solving
    dat_file_path = _select_data_for_solve(input_dir, log_dir)
    ampl_snippet = _safe_read(os.path.join(input_dir, "data.dat"), "").strip()

    instruction = _read_prompt("ampl_instruction.txt")
    guideline  = _read_prompt("ampl_guideline.txt")
    ampl_refine_template = _read_prompt(
        "ampl_refinement_prompt.txt",
        "Refine the AMPL model based on this error:\n{ERROR_MESSAGE}\n"
        "Return only the model section."
    )

    description_target = target.get("description", "")
    objective_target = target.get("objective", "")
    constraints_target = target.get("constraints", [])
    parameters_target = target.get("parameter", [])


    initial_prompt = (
        f"Instructions:\n{instruction}\n\n"
        f"Ampl Guideline:\n{guideline}\n\n"
        f"Problem Description:\n{description_target}\n\n"
        f"Parameters:\n{json.dumps(parameters_target, indent=2)}\n\n"
        f"Objective:\n{objective_target}\n\n"
        f"Constraints:\n{json.dumps(constraints_target, indent=2)}\n\n"
        f"Run Number: {run_number}\n\n"
        "Generate only the AMPL model file (.mod). "
        "Output raw AMPL code. No data section."
    )
    if ampl_snippet:
        initial_prompt += (
                "\n\nREAD-ONLY DATA CONTEXT: Generate a fully-functional AMPL model (.mod) **and** matching data file (.dat) for the optimization problem. (do not reprint; do not generate a .dat):\n"
                + ampl_snippet[:4000]  # guard size if large
        )

    llm = get_llm(model)

    messages = []
    conversation_log = []
    llm_call_count = 0

    def send_message(prompt):
        nonlocal llm_call_count
        conversation_log.append({"role": "user", "content": prompt})
        messages.append(HumanMessage(content=prompt))
        content = llm_call(llm, messages, use_logprobs=use_logprobs, log_dir=log_dir)
        llm_call_count += 1
        conversation_log.append({"role": "assistant", "content": content})
        messages.append(AIMessage(content=content))
        return content

    # Initial model generation
    generated_code = send_message(initial_prompt)

    model_part_only = generated_code.split("---DATA---")[0].replace("---MODEL---", "").strip()
    model_part_only = clean_code(model_part_only, is_model=True)

    mod_file_path = os.path.join(log_dir, "model.mod")
    with open(mod_file_path, "w", encoding="utf-8") as mod_file:
        mod_file.write(model_part_only)

    # Solve using the selected user data
    solution_result = solve_ampl(log_dir, model_file='model.mod', data_file=os.path.basename(dat_file_path), solution_file='initial_solution.txt')
    initial_solution = parse_ampl_solution(solution_result)
    final_solution_path = solution_result
    refinement_attempts = 0


    if refinement and initial_solution["status"] in ["error", "syntax_error", "unknown"]:
        for attempt in range(1, max_refine + 1):
            refinement_attempts = attempt
            with open(solution_result, 'r', encoding="utf-8") as f:
                error_message = f.read()

            refine_prompt = ampl_refine_template.format(
                PREVIOUS_MODEL=model_part_only,
                ERROR_MESSAGE=error_message,
                ATTEMPT=attempt,
                AMPL_GUIDELINE_GIST=guideline,
                DESCRIPTION=description_target,
                PARAMETERS=json.dumps(parameters_target, indent=2),
                OBJECTIVE=objective_target,
                CONSTRAINTS=json.dumps(constraints_target, indent=2)
            )
            if ampl_snippet:
                refine_prompt += (
                        "\n\nREAD-ONLY DATA CONTEXT (do not reprint; do not generate a .dat):\n"
                        + ampl_snippet[:4000]  # guard size if large
                )

            generated_code = send_message(refine_prompt)
            corrected_model = generated_code.split("---DATA---")[0].replace("---MODEL---", "").strip()
            corrected_model = clean_code(corrected_model, is_model=True)

            refine_mod_path = os.path.join(log_dir, f'model-refinement_attempt_{attempt}.mod')
            with open(refine_mod_path, 'w', encoding="utf-8") as f:
                f.write(corrected_model)

            refine_solution_path = solve_ampl(
                log_dir,
                model_file=os.path.basename(refine_mod_path),
                data_file=os.path.basename(dat_file_path),
                solution_file=f'refinement_solution_attempt_{attempt}.txt'
            )
            refine_solution = parse_ampl_solution(refine_solution_path)

            if refine_solution["status"] not in ["error", "syntax_error", "unknown"]:
                final_solution_path = refine_solution_path
                break
            solution_result = refine_solution_path
            model_part_only = corrected_model  # carry forward best-known model

    final_solution_file = os.path.join(log_dir, 'final_solution.txt')
    with open(final_solution_path, 'r', encoding="utf-8") as src, open(final_solution_file, 'w', encoding="utf-8") as dst:
        dst.write(src.read())

    with open(os.path.join(log_dir, "conversation_log.json"), "w", encoding="utf-8") as f:
        json.dump(conversation_log, f, indent=2)

    print(f"Number of LLM calls: {llm_call_count}")

    return mod_file_path, dat_file_path, final_solution_file, refinement_attempts, llm_call_count

def main():
    parser = argparse.ArgumentParser(
        description="Generate AMPL model and solve using user data."
    )
    parser.add_argument("--input_dir", required=True, help="Path to the problem instance directory")
    parser.add_argument("--model", default="gpt-4-1106-preview", help="Model name")
    parser.add_argument("--logprob", action="store_true", help="Turn on log probability output")
    parser.add_argument("--logs", default=None, help="Directory to save output logs")
    parser.add_argument("--refinement", action="store_true", help="Turn on refinement loop")
    parser.add_argument("--max_refine", type=int, default=1, help="Maximum number of refinement attempts")
    args = parser.parse_args()

    log_dir = args.logs or os.path.join("logs", f"log_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)

    mod_path, dat_path, final_solution_path, refinement_attempts, llm_call_count = generate_ampl_files(
        args.input_dir, args.model, log_dir, args.logprob, run_number=1,
        refinement=args.refinement, max_refine=args.max_refine
    )

    print("AMPL files generated:")
    print("Model file:", mod_path)
    print("Data file used for solve:", dat_path)
    print("Final solution file:", final_solution_path)
    print("Refinement attempts:", refinement_attempts)
    print("LLM calls:", llm_call_count)

if __name__ == "__main__":
    main()

