import os
import json
import argparse
import time
from llm_utils import CONFIG, get_llm, llm_call
from langchain_core.messages import HumanMessage, AIMessage

PROMPT_TEMPLATE = r"""Extract the following information from the given natural language optimization problem description.

Your tasks are:
1. Identify and extract all parameters. For each parameter, provide:
   - **symbol**: the parameter's symbol.
   - **definition**: a brief explanation of what it represents.
   - **dimension**: a list indicating the dimensions (if any).
2. Rewrite the problem description so that every parameter is referenced using symbolic notation (e.g., \param{{symbol}} or \param{{symbol}}_i).
3. Extract a list of all constraints found in the description.
4. Extract the objective of the optimization problem.
5. List the decision variables. For each variable, provide:
    - **symbol**: the variable's symbol.
    - **definition**: a brief explanation of what it represents.
    - **dimension**: a list indicating the dimensions (if any).

Return your answer in JSON format with the following structure:

{{
  "parameter": [{{...}}],
  "description": "...",
  "constraints": ["..."],
  "objective": "...",
  "variables": [{{...}}]
}}

---

Example:

Problem:
Our company currently ships products from five plants to four warehouses. It is considering closing some plants to reduce costs. Which plant(s) should our company close to minimize transportation while ensuring warehouse demand is met within each plant's capacity?

Data:
{{
  "demand": [15, 18, 14, 20],
  "capacity": [20, 22, 17, 19, 18],
  "fixedCosts": [12000, 15000, 17000, 13000, 16000],
  "transCosts": [
    [4000, 2000, 3000, 2500, 4500],
    [2500, 2600, 3400, 3000, 4000],
    [1200, 1800, 2600, 4100, 3000],
    [2200, 2600, 3100, 3700, 3200]
  ]
}}

Answer:
{{
  "parameter": [
    {{
      "symbol": "demand",
      "definition": "Demand at each warehouse",
      "dimension": ["NumWarehouses"]
    }},
    {{
      "symbol": "capacity",
      "definition": "Maximum production capacity at each plant",
      "dimension": ["NumPlants"]
    }},
    {{
      "symbol": "fixedCosts",
      "definition": "Fixed cost to keep each plant open",
      "dimension": ["NumPlants"]
    }},
    {{
      "symbol": "transCosts",
      "definition": "Transportation cost from each plant to each warehouse",
      "dimension": ["NumWarehouses", "NumPlants"]
    }}
  ],
  "description": "Our company has \param{{NumPlants}} plants and \param{{NumWarehouses}} warehouses. The fixed cost for keeping plant i open is \param{{fixedCosts}}_i. Each plant i has capacity \param{{capacity}}_i. The cost of transporting from plant i to warehouse j is \param{{transCosts}}_{{j,i}}. Each warehouse j has a demand of \param{{demand}}_j. Decide which plants to close and how much to ship to minimize total cost.",
  "constraints": [
    "Each warehouse's demand must be fully satisfied by the sum of products shipped from all open plants",
    "Total shipment from each plant must not exceed its capacity if it remains open",
    "Binary variable to indicate whether a plant is open or closed"
  ],
  "objective": "Minimize the total cost, which includes fixed costs for open plants and transportation costs from plants to warehouses",
  "variables": [
    {{
      "symbol": "open[i]",
      "definition": "1 if plant i is open, 0 otherwise",
      "dimension": []
    }},
    {{
      "symbol": "ship[i][j]",
      "definition": "Quantity shipped from plant i to warehouse j",
      "dimension": []
    }}
  ]
}}

---

Now analyze the following problem:

Problem Description:
{description}

Data:
{data}
"""

def generate_prompt(description: str, data: str) -> str:
    return PROMPT_TEMPLATE.format(description=description, data=data)

def process_prompt(prompt: str, model: str, log_dir: str = None, use_logprobs: bool = False, run_number: int = None) -> dict:
    """
    Uses the configured LLM to produce structured JSON. Retries up to 3 times on JSON parsing errors.
    """
    attempts = 3

    llm = get_llm(model)

    from langchain_core.messages import HumanMessage, AIMessage
    messages = []

    while attempts > 0:
        try:
            messages.append(HumanMessage(content=prompt))
            content = llm_call(llm, messages, use_logprobs=use_logprobs, log_dir=log_dir)
            # Extract JSON payload if fenced
            decision = content.strip()
            if "```json" in decision:
                decision = decision.split("```json")[1].split("```")[0]
            # Normalize a few characters
            decision = decision.replace("\\u2212", "-").replace("\u2212", "-")

            result = json.loads(decision)
            return result
        except Exception as e:
            attempts -= 1
            if attempts == 0:
                raise
    # Fallback, should not reach
    return {"parameter": [], "description": "", "constraints": [], "objective": "", "variables": []}

def main():
    parser = argparse.ArgumentParser(
        description="Convert a natural language optimization problem and data into structured JSON."
    )
    parser.add_argument("--description", required=True, help="Path to problem description .txt file")
    parser.add_argument("--data", required=True, help="Path to data .json or .csv file")
    parser.add_argument("--model", default="gpt-4-1106-preview", help="Model name")
    parser.add_argument("--logprob", action="store_true", help="Enable log probability output")
    parser.add_argument("--logs", default=None, help="Directory to save output logs")
    parser.add_argument("--run_number", type=int, default=None, help="Run number for seeding the LLM call")
    args = parser.parse_args()

    log_dir = args.logs or os.path.join("logs", f"log_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)

    with open(args.description, "r", encoding="utf-8") as f:
        description_text = f.read()
    with open(args.data, "r", encoding="utf-8") as f:
        data_text = f.read()

    prompt = generate_prompt(description_text, data_text)
    structured_output = process_prompt(prompt, args.model, log_dir, args.logprob, run_number=args.run_number)

    print(json.dumps(structured_output, indent=2))

    output_path = os.path.join(log_dir, "structured_description.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured_output, f, indent=2)
    print(f"Saved structured output to {output_path}")

if __name__ == "__main__":
    main()
