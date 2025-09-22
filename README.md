<img src="images/logo.png" alt="EXEOS Logo" width="130" align="left" />

# EXEOS: EXtraction and Error-guided refinement of Optimization Specifications

EXEOS is an LLM-based pipeline designed to generate mathematical optimization specifications from natural language (NL) problem statements. It derives both AMPL models (a domain-specific language for optimization) and Python code (using Gurobi) from NL descriptions, iteratively refining them using solver feedback to improve executability and correctness. This project is based on the paper *"Models or Code? Evaluating the Quality of LLM-Generated Specifications: A Case Study in Optimization at Kinaxis"*, which evaluates the LLM-generated models in DSLs like AMPL and direct code in general-purpose languages like Python. The evaluation uses a public optimization dataset and real-world supply-chain cases from [Kinaxis](https://www.kinaxis.com), showing that AMPL is competitive with—and sometimes surpasses—Python in quality.
The pipeline shifts the cost balance in model-driven engineering (MDE) by automating the generation of structured artifacts from text, enabling domain experts to work in NL while preserving modeling benefits.

## Approach Overview
The EXEOS approach consists of four main stages, as outlined in Figure 4 of the paper:
1. **Structure and Add Metadata**: Identifies the main components of the optimization problem (objectives, parameters, variables, constraints) from the NL description, organizes them into a structured format, and extracts metadata (e.g., symbols, definitions, dimensions) for parameters and variables. The original description is rewritten with markup references to these symbols.
2. **Transform Data**: Processes user-supplied tabular data for parameter values, using metadata from Step 1, into a solver-ready format (AMPL-compatible .dat file or JSON for Python).
3. **Generate Formal Specification**: Transforms the structured description from Step 1 into a formal optimization specification using an LLM. This step handles initial generation (with prompts including syntax rules, few-shot examples, and the structured output) or refinement (extending prompts with prior specifications and solver feedback if errors occur).
4. **Solve the Optimization Problem**: Uses an optimization solver (e.g., Gurobi) with the specification from Step 3 and data file from Step 2 to compute a solution. If compilation or runtime errors arise, initiates a refinement loop back to Step 3 until solved or an iteration limit is reached.
   
![Approach Figure](images/approach.png)

## Project Structure

- **supplementary/**: Contains prompt outlines and results PDFs.
- **data/**: Public dataset with 60 NL optimization problems from textbooks.
  - `description.txt`: NL problem description with embedded data references.
  - `description2.txt`: NL problem description without explicit data (our approach).
  - Data files in JSON format (`data.json`) and AMPL format (`ampl-data.txt`).
- **code/**: Core implementation files.
  - `app.py`: Main pipeline orchestrator (NL → Structured → AMPL/Python → Solve).
  - `nl_to_structured.py`: Handles NL to structured JSON extraction.
  - `ampl_generator.py`: Generates and refines AMPL models/data.
  - `python_generator.py`: Generates and refines Python (Gurobi) code.
  - `requirements.txt`: Python dependencies.
- **results/**: Contains the outcomes of experiments.


  ## Prerequisites
- Python 3.8+.
- [AMPL](https://ampl.com/): Requires a license (community edition available for small problems; full license for larger ones). Install via `amplpy`.
- [Gurobi](https://www.gurobi.com/): Requires a license (academic/free trials available). Install via `gurobipy`.
- LLM APIs: OpenAI (default) or Google Vertex AI (Gemini). Requires API keys.
- Dependencies listed in `requirements.txt`.

Note: Some tools like AMPL and Gurobi require licenses for full functionality. LLMs need API keys—configure them in `llm_utils.py` or via environment variables.

## Installation and Configuration

1. Clone the repository:
-`git clone https://github.com/neayoughi/EXEOS.git`
-`cd EXEOS`

3. Install dependencies: `pip install -r requirements.txt`
4. Configure APIs and Licenses:
- **OpenAI API Key**: Set `OPENAI_API_KEY` in your environment or update `CONFIG["openai_api_key"]` in `llm_utils.py`.
- **Google Vertex AI (Optional)**: Set `project_id` and `location` in `get_llm` calls (e.g., in `ampl_generator.py`, `python_generator.py`).
- **AMPL License**: Obtain from [AMPL website](https://ampl.com/) and configure in your environment (e.g., via `AMPL_LICENSE` or community edition setup).
- **Gurobi License**: Obtain from [Gurobi website](https://www.gurobi.com/) and set `GRB_LICENSE_FILE` or use academic licensing.
- Ensure `prompt/` directory exists with templates (or set `EXEOS_PROMPT_DIR` env var).
## Running the Application and Usage

Run the pipeline via `app.py` to process an NL optimization problem. It generates structured output, AMPL/Python specifications, solves them, and refines on errors.

### Basic Usage
`python app.py --nl_file path/to/description.txt --data_file path/to/data.json [path/to/ample-data.txt] `
- If providing AMPL data, pass it as the second file in `--data_file` (optional).

### CLI Options
- `--nl_file`: Path to .txt file with NL optimization description.
- `--data_file`: One or two files: JSON data.
- `--model` (default: "gpt-4-1106-preview"): LLM model (e.g., "gpt-4o", "vertex-gemini").
- `--refinement`: Enable iterative refinement on errors (default: disabled).
- `--maxtry` (default: 2): Max refinement attempts.

### Example
`python app.py --nl description.txt --data_file data.json data.dat --refinement --maxtry 3`
Outputs are saved in `logs/run_<timestamp>/` (e.g., structured JSON, AMPL/Python files, solutions). The console prints a JSON summary of results.



