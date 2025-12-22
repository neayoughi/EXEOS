import json
import os
import subprocess
import shutil
import time
import argparse
import re
from llm_utils import get_llm, llm_call
from langchain_core.messages import HumanMessage, AIMessage

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

def _maybe_structure(input_dir, model, log_dir, use_structuring):
    it_path = os.path.join(input_dir, "input_targets.json")
    if not use_structuring and os.path.exists(it_path):
        with open(it_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if use_structuring:
        struct_log = os.path.join(log_dir, "struct")
        os.makedirs(struct_log, exist_ok=True)
        desc_path = os.path.join(input_dir, "description.txt")
        data_path = os.path.join(input_dir, "data.json")
        subprocess.run(
            ["python", "nl_to_structured.py",
             "--description", desc_path,
             "--data", data_path,
             "--model", model,
             "--logs", struct_log],
            check=True,
            cwd="."
        )
        out_json = os.path.join(struct_log, "structured_description.json")
        with open(out_json, "r", encoding="utf-8") as f:
            s = json.load(f)
        targets = {
            "background": "",
            "description": s.get("description", _safe_read(desc_path, "")),
            "objective": s.get("objective", ""),
            "constraints": s.get("constraints", []),
            "parameter": s.get("parameter", [])
        }
        with open(it_path, "w", encoding="utf-8") as f:
            json.dump(targets, f, indent=2)
        return targets

    desc_text = _safe_read(os.path.join(input_dir, "description.txt"), "")
    return {"background": "", "description": desc_text, "objective": "", "constraints": [], "parameter": []}

def clean_code(code):
    code = re.sub(r'```python\s*\n|```(?:\s*\n|$)', '', code)
    lines = code.splitlines()
    cleaned_lines = [line.rstrip() for line in lines if line.strip()]
    return '\n'.join(cleaned_lines)

def parse_solution(output_text, error_text):
    solution = {
        "status": "unknown",
        "objective_value": None,
        "variables": None,
        "error_message": "",
        "error_type": "None",
        "runtime": 0,
        "compile": 0
    }

    if "SyntaxError" in error_text:
        solution["status"] = "syntax_error"
        solution["error_message"] = error_text
        solution["error_type"] = "Compilation"
        solution["compile"] = 1
        return solution

    if error_text:
        solution["status"] = "error"
        solution["error_message"] = error_text
        solution["error_type"] = "Runtime"
        solution["runtime"] = 1
        return solution

    lines = output_text.strip().split("\n")
    for i, line in enumerate(lines):
        if line.startswith("Optimal Objective Value:"):
            try:
                solution["status"] = "optimal"
                solution["objective_value"] = float(line.split(":", 1)[1].strip())
                solution["variables"] = {}
                for var_line in lines[i + 1:]:
                    var_line = var_line.strip()
                    if var_line and ":" in var_line:
                        parts = var_line.rsplit(":", 1)
                        var_name = parts[0].strip()
                        var_value = float(parts[1].strip())
                        solution["variables"][var_name] = var_value
                break
            except (ValueError, IndexError):
                solution["status"] = "error"
                solution["error_message"] = "Failed to parse objective or variable values"
                solution["error_type"] = "Runtime"
                solution["runtime"] = 1
        elif line == "No feasible solution found.":
            solution["status"] = "infeasible"
            break

    return solution

def generate_gurobi_code(input_dir, model, log_dir, use_logprobs, run_number=None, refinement=False, max_refine=1, structure=False):
    target = _maybe_structure(input_dir, model, log_dir, use_structuring=structure)

    with open(os.path.join(input_dir, "data.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    py_snippet = _safe_read(os.path.join(input_dir, "data.json"), "").strip()

    guidelines  = _read_prompt("gurobi_guidelines.txt", "")
    instruction = _read_prompt(
        "gurobi_instruction.txt",
        "Write a complete Python script using Gurobi. Load JSON from 'input.json'. "
        "Build model, variables, objective, constraints. Optimize. "
        "Print 'Optimal Objective Value: <value>' then one '<var>: <value>' per line."
    )
    example     = _read_prompt("example.txt", "# Example omitted.")
    refine_tmpl = _read_prompt(
        "Python_refinement_prompt.txt",
        "The Python optimization script failed.\n\n"
        "Problem description:\n{DESCRIPTION}\n\n"
        "Parameters:\n{PARAMETERS}\n\n"
        "Objective:\n{OBJECTIVE}\n\n"
        "Constraints:\n{CONSTRAINTS}\n\n"
        "Previous model:\n{PREVIOUS_MODEL}\n\n"
        "Error log:\n{ERROR_MESSAGE}\n\n"
        "Attempt {ATTEMPT}\n\n"
        "{GUIDELINE}\n"
    )

    description_target = target.get("description", "")
    objective_target = target.get("objective", "")
    constraints_target = target.get("constraints", [])
    parameters_target = target.get("parameter", [])

    initial_prompt = (
        f"Instructions:\n{instruction}\n\n"
        f"Gurobi Guidelines:\n{guidelines}\n\n"
        f"Example Gurobi Code:\n{example}\n\n"
        f"Problem Description:\n{description_target}\n\n"
        f"Parameters:\n{json.dumps(parameters_target, indent=2)}\n\n"
        f"Objective:\n{objective_target}\n\n"
        f"Constraints:\n{json.dumps(constraints_target, indent=2)}\n\n"
        "Generate a complete Python script using the Gurobi Python API to formulate and solve the optimization problem. "
        "Output only the raw Python code without code fences. "
        "The script must load data from 'input.json', define the model, variables, objective, constraints, optimize, "
        "print results as 'Optimal Objective Value: <value>' and '<var_name>: <value>' lines, and export an MPS file."
    )
    if py_snippet:
        initial_prompt += (
                "\n\nREAD-ONLY DATA CONTEXT (do not reprint; do not generate a .json):\n"
                + py_snippet[:4000]  # guard size if large
        )

    llm = get_llm(model)
    messages = []
    conversation_log = []
    llm_call_count = 0

    def send_message(prompt):
        nonlocal llm_call_count, conversation_log
        conversation_log.append({"role": "user", "content": prompt})
        messages.append(HumanMessage(content=prompt))
        # content = llm_call(llm, messages, use_logprobs=use_logprobs, log_dir=log_dir,use_thinking=True)
        content = llm_call(llm, messages, use_logprobs=use_logprobs, log_dir=log_dir)
        llm_call_count += 1
        conversation_log.append({"role": "assistant", "content": content})
        messages.append(AIMessage(content=content))
        return content

    generated_code = send_message(initial_prompt)
    generated_code = clean_code(generated_code)

    code_path = os.path.join(log_dir, "generated_code.py")
    if refinement:
        code_path = os.path.join(log_dir, "generated_code_attempt_0.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(generated_code)

    response_log_path = os.path.join(log_dir, f"response_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(response_log_path, "w", encoding="utf-8") as f:
        f.write(generated_code)

    shutil.copy(os.path.join(input_dir, "data.json"), os.path.join(log_dir, "input.json"))

    solution = None
    attempt = 0
    max_attempts = max_refine if refinement else 0
    while attempt <= max_attempts:
        current_code_path = code_path if attempt == 0 else os.path.join(log_dir, f"generated_code_attempt_{attempt}.py")
        try:
            result = subprocess.run(
                ["python3", os.path.basename(current_code_path)],
                cwd=log_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60
            )
            output_text = result.stdout if result.returncode == 0 else ""
            error_text = "" if result.returncode == 0 else result.stderr
            solution = parse_solution(output_text, error_text)

            if solution["status"] in ["error", "syntax_error"] and refinement and attempt < max_refine:
                refine_prompt = refine_tmpl.format(
                    DESCRIPTION=description_target,
                    PARAMETERS=json.dumps(parameters_target, indent=2),
                    OBJECTIVE=objective_target,
                    CONSTRAINTS=json.dumps(constraints_target, indent=2),
                    PREVIOUS_MODEL=generated_code,
                    ERROR_MESSAGE=error_text or "Unknown error",
                    ATTEMPT=attempt + 1,
                    GUIDELINE=guidelines,
                    )
                if py_snippet:
                    refine_prompt += (
                        "\n\nREAD-ONLY DATA CONTEXT (do not reprint; do not generate a .json):\n"
                        + py_snippet[:4000]
                        )
                generated_code = send_message(refine_prompt)
                generated_code = clean_code(generated_code)
                new_code_path = os.path.join(log_dir, f"generated_code_attempt_{attempt + 1}.py")
                with open(new_code_path, "w", encoding="utf-8") as f:
                    f.write(generated_code)
                attempt += 1
                continue

            # if solution["status"] in ["error", "syntax_error"] and refinement and attempt < max_refine:
            #     refine_prompt = refine_tmpl.format(
            #         ERROR=error_text or "Unknown error",
            #         CODE=generated_code
            #     )
            #     if py_snippet:
            #         initial_prompt += (
            #                 "\n\nREAD-ONLY DATA CONTEXT (do not reprint; do not generate a .json):\n"
            #                 + py_snippet[:4000]  # guard size if large
            #         )
            #     generated_code = send_message(refine_prompt)
            #     generated_code = clean_code(generated_code)
            #     new_code_path = os.path.join(log_dir, f"generated_code_attempt_{attempt + 1}.py")
            #     with open(new_code_path, "w", encoding="utf-8") as f:
            #         f.write(generated_code)
            #     attempt += 1
            #     continue

            

            break
        except subprocess.TimeoutExpired:
            solution = {"status": "error", "error_message": "Execution timed out after 60 seconds", "error_type": "Runtime", "runtime": 1}
            break
        except Exception as e:
            solution = {"status": "error", "error_message": str(e), "error_type": "Runtime", "runtime": 1}
            break

    solution_json_path = os.path.join(log_dir, "solution.json")
    with open(solution_json_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2)

    with open(os.path.join(log_dir, "conversation_log.json"), "w", encoding="utf-8") as f:
        json.dump(conversation_log, f, indent=2)

    final_code_path = (
    os.path.join(log_dir, f"generated_code_attempt_{attempt}.py")
    if refinement
    else code_path
    )
    return final_code_path, solution_json_path, attempt, llm_call_count

    # return code_path, solution_json_path, attempt, llm_call_count

def main():
    parser = argparse.ArgumentParser(description="Generate Gurobi code and solve optimization problem")
    parser.add_argument("--input_dir", required=True, help="Path to input directory")
    parser.add_argument("--model", default="gpt-4-1106-preview", help="LLM model name")
    parser.add_argument("--log_dir", required=True, help="Directory to save logs and outputs")
    parser.add_argument("--use_logprobs", action="store_true", help="Enable log probabilities")
    parser.add_argument("--refinement", action="store_true", help="Turn on refinement loop")
    parser.add_argument("--max_refine", type=int, default=1, help="Maximum number of refinement attempts")
    parser.add_argument("--structure", action="store_true", help="First structure the NL description")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    code_path, solution_path, attempts, llm_calls = generate_gurobi_code(
        args.input_dir, args.model, args.log_dir, args.use_logprobs,
        refinement=args.refinement, max_refine=args.max_refine, structure=args.structure
    )
    print(f"Generated code saved to: {code_path}")
    print(f"Solution saved to: {solution_path}")
    print(f"Refinement attempts: {attempts}")
    print(f"LLM calls: {llm_calls}")

if __name__ == "__main__":
    main()
