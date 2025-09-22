import argparse
import os
import json
import subprocess
import textwrap
import time
import re
import sys
import csv
from llm_utils import get_llm, CONFIG
from langchain_core.messages import HumanMessage, AIMessage
from amplpy import AMPL, AMPLException
from langchain_openai import ChatOpenAI
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
    Handles optimal solutions, license errors, syntax errors, and other exceptions.
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

def parse_ampl_response(response):
    """
    Parse AMPL response to separate model and data parts.
    """
    if "---MODEL---" in response and "---DATA---" in response:
        model_part = response.split("---MODEL---")[1].split("---DATA---")[0].strip()
        data_part = response.split("---DATA---")[1].strip()
    else:
        model_part = response.split("---DATA---")[0].strip() if "---DATA---" in response else response.strip()
        data_part = response.split("---DATA---")[1].strip() if "---DATA---" in response else ""
    return model_part, data_part

# Updated solve_ampl function
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

def generate_ampl_files(input_dir, model, log_dir, use_logprobs=False, run_number=None, debug=False, maxtry=1, self_reflection=False, max_semantic_try=1):
    with open(os.path.join(input_dir, "input_targets.json"), "r", encoding="utf-8") as f:
        target = json.load(f)
    with open(os.path.join(input_dir, "data.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    ampl_data = _safe_read(os.path.join(input_dir, "ampl_data.txt"), "")
    nl_description = _safe_read(os.path.join(input_dir, "description.txt"), "")

    guideline   = _read_prompt("ampl_guideline.txt")
    instruction = _read_prompt("ampl_instruction.txt")
    refinement_template       = _read_prompt("refinement_prompt.txt",       "Fix the AMPL model and data based on this error:\n{ERROR_MESSAGE}\nReturn only ---MODEL--- then ---DATA---.")


    description_target = target.get("description", "")
    objective_target = target.get("objective", "")
    constraints_target = target.get("constraints", [])
    parameters_target = target.get("parameter", [])

    initial_prompt = (
        f"Instructions:\n{instruction}\n\n"
        f"Input Data:\n{json.dumps(data, indent=2)}\n\n"
        f"Ampl Data:\n{ampl_data}\n\n"
        f"Problem Description:\n{description_target}\n\n"
        f"Parameters:\n{json.dumps(parameters_target, indent=2)}\n\n"
        f"Objective:\n{objective_target}\n\n"
        f"Constraints:\n{json.dumps(constraints_target, indent=2)}\n\n"
        f"Run Number: {run_number}\n\n"
        "Generate the complete AMPL model file (.mod) and data file (.dat). "
        "Output the raw AMPL code without any code block markers. "
        "The output must have two sections: first the model code, then the data code, "
        "separated by the exact labels '---MODEL---' and '---DATA---'."
    )

    llm = get_llm(model, api_key=CONFIG["openai_api_key"], project_id="", location="") # Your GCP project_id="", location=""
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

    generated_code = send_message(initial_prompt)
    model_part, data_part = parse_ampl_response(generated_code)

    model_part = clean_code(model_part, is_model=True)
    data_part = clean_code(data_part, is_model=False)
    mod_file_path = os.path.join(log_dir, "model.mod")
    dat_file_path = os.path.join(log_dir, "data.dat")
    with open(mod_file_path, "w", encoding="utf-8") as mod_file:
        mod_file.write(model_part)
    with open(dat_file_path, "w", encoding="utf-8") as dat_file:
        dat_file.write(data_part)

    solution_result = solve_ampl(log_dir, model_file='model.mod', data_file='data.dat', solution_file='initial_solution.txt')
    initial_solution = parse_ampl_solution(solution_result)
    final_solution_path = solution_result
    refinement_attempts = 0

    if refinement and initial_solution["status"] in ["error", "syntax_error", "unknown"]:
        for attempt in range(1, maxtry + 1):
            refinement_attempts = attempt
            with open(solution_result, 'r', encoding="utf-8") as f:
                error_message = f.read()

            refinement_prompt = refinement_template.format(
                PREVIOUS_MODEL=model_part,
                PREVIOUS_DATA=data_part,
                ERROR_MESSAGE=error_message,
                ATTEMPT=attempt,
                AMPL_GUIDELINE=guideline,
                INPUT_DATA=json.dumps(data, indent=2),
                AMPL_DATA=ampl_data,
                DESCRIPTION=description_target,
                PARAMETERS=json.dumps(parameters_target, indent=2),
                OBJECTIVE=objective_target,
                CONSTRAINTS=json.dumps(constraints_target, indent=2)
            )

            generated_code = send_message(refinement_prompt)
            corrected_model, corrected_data = parse_ampl_response(generated_code)

            corrected_model = clean_code(corrected_model, is_model=True)
            corrected_data = clean_code(corrected_data, is_model=False)
            refinement_mod_path = os.path.join(log_dir, f'model-refinement_attempt_{attempt}.mod')
            refinement_dat_path = os.path.join(log_dir, f'data-refinement_attempt_{attempt}.dat')
            with open(refinement_mod_path, 'w', encoding="utf-8") as f:
                f.write(corrected_model)
            with open(refinement_dat_path, 'w', encoding="utf-8") as f:
                f.write(corrected_data)

            refinement_solution_path = solve_ampl(log_dir, model_file=f'model-refinement_attempt_{attempt}.mod', data_file=f'data-refinement_attempt_{attempt}.dat', solution_file=f'refinement_solution_attempt_{attempt}.txt')
            refinement_solution = parse_ampl_solution(refinement_solution_path)

            if refinement_solution["status"] not in ["error", "syntax_error", "unknown"]:
                final_solution_path = refinement_solution_path
                break
            solution_result = refinement_solution_path



    final_solution_file = os.path.join(log_dir, 'final_solution.txt')
    with open(final_solution_path, 'r', encoding="utf-8") as src, open(final_solution_file, 'w', encoding="utf-8") as dst:
        dst.write(src.read())

    with open(os.path.join(log_dir, "conversation_log.json"), "w", encoding="utf-8") as f:
        json.dump(conversation_log, f, indent=2)

    print(f"Number of LLM calls: {llm_call_count}")

    return mod_file_path, dat_file_path, final_solution_file, refinement_attempts, llm_call_count

def main():
    parser = argparse.ArgumentParser(
        description="Generate AMPL model and data files for an optimization problem."
    )
    parser.add_argument("--input_dir", required=True, help="Path to the problem instance directory")
    parser.add_argument("--model", default="gpt-4-1106-preview", help="Model name (e.g., gpt-4o, vertex-gemini)")
    # parser.add_argument("--logprob", action="store_true", help="Enable log probability output")
    parser.add_argument("--logs", default=None, help="Directory to save output logs")
    parser.add_argument("--refinement", action="store_true", help="Enable refinement mode")
    parser.add_argument("--maxtry", type=int, default=1, help="Maximum number of refinement attempts")
  =
    args = parser.parse_args()

    log_dir = args.logs or os.path.join("logs", f"log_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)

    mod_path, dat_path, final_solution_path, refinement_attempts, llm_call_count = generate_ampl_files(
        args.input_dir, args.model, log_dir, run_number=1,
        refinement=args.refinement, maxtry=args.maxtry
    )

    print("AMPL files generated:")
    print("Model file:", mod_path)
    print("Data file:", dat_path)
    print("Final solution file:", final_solution_path)
    print("refinement attempts:", refinement_attempts)
    print("LLM calls:", llm_call_count)

if __name__ == "__main__":
    main()
