import os
import time
import re
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import json
import vpp

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def load_shuffled_csv_files(folder_path, file_configs):
    combined_lines = []

    for file_name, num_rows in file_configs.items():
        full_path = os.path.join(folder_path, file_name)

        if not os.path.exists(full_path):
            print(f"⚠️ Warning: File {file_name} not found, skipping.")
            continue

        try:
            df = pd.read_csv(full_path)

            sample_size = min(num_rows, len(df))
            df_sampled = df.sample(n=sample_size, random_state=42)

            df_sampled = df_sampled.sort_values(by=df_sampled.columns[0])

            combined_lines.append(f"--- Random Sampled Data from {file_name} ({sample_size} rows) ---")
            combined_lines.append(df_sampled.to_string(index=False))
            combined_lines.append("\n")

            print(f"🎲 Successfully sampled {sample_size} random rows from {file_name}")
        except Exception as e:
            print(f"❌ Error reading {file_name}: {e}")

    return "\n".join(combined_lines)

data_folder = "../data/final_data"

configs = {
    "5_digits.csv": 50,
    "6_digits.csv": 50
}

data_string = load_shuffled_csv_files(data_folder, configs)

def get_next_filter_id(file_path):
    if not os.path.exists(file_path):
        return 1

    with open(file_path, "r") as f:
        content = f.read()
        matches = re.findall(r"filter_(\d+)_", content)
        if not matches:
            return 1
        return max(int(m) for m in matches) + 1

STRATEGY_FILE = "../output/filter_strategies.py"

def load_history_context(ledger_path="../output/ledger.json"):
    """
    Loads the history of previously generated strategies from a JSON ledger.
    This injects 'long-term memory' into the LLM prompt to prevent cyclic repetition.
    """
    empty_message = "[History Context: The strategy library is currently empty. You have complete freedom to propose your first mathematical trick.]\n"

    if not os.path.exists(ledger_path):
        return empty_message

    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)

        if not ledger:
            return empty_message

        history_text = "[Explored Strategies Library - CRITICAL: Do NOT duplicate the mathematical logic of the following filters]\n"

        for item in ledger:
            # Safely extract fields with fallbacks
            strategy_name = item.get("name", "Unnamed_Strategy")
            description = item.get("description", "No description provided.")

            history_text += f"- Filter Name: {strategy_name}\n"
            history_text += f"  Mathematical Core: {description}\n\n"

        return history_text

    except json.JSONDecodeError:
        print(f"⚠️ Warning: {ledger_path} is not a valid JSON file. Returning empty history.")
        return empty_message
    except Exception as e:
        print(f"⚠️ Warning: Failed to read {ledger_path}: {e}")
        return empty_message

# 1. System Prompt: Setting the expert persona and strict boundaries
def get_system_prompt(next_id):
    # 1. Dynamically load the memory constraint
    history_context = load_history_context()

    # 2. Assemble the prompt
    return f"""You are an expert computational number theorist building an automated Pomerance search engine.

[Data Context]
Here is a sample dataset of primes and their valid Pomerance parameters (p, A, x0):
{data_string}

{history_context}

[Task]
Autonomously discover a NOVEL statistical or algebraic constraint to prune the search space for 'A' or 'x0'. 
Your output must be a single Python function `apply_filter(A, x0, p)` that returns True (keep) or False (discard).

[Strict Engineering Constraints]
1. NAMING: The function MUST be named `filter_{next_id:02d}_[your_descriptive_name]` (e.g., `filter_{next_id:02d}_legendre_bias`).
2. MODULARITY: Do NOT include elliptic curve doubling loops or GCD verification. Assume all heavy math primitives are executed elsewhere. Output pure filtering logic only.
3. MATHEMATICAL RIGOR: 
   - Deterministic filters: Must guarantee a 0% False Negative rate.
   - Probabilistic/Heuristic filters: Must explicitly state the expected pruning rate and any collision risks in the docstring.
4. DOCUMENTATION: Your function must begin with a docstring formatted exactly like this:
   \"\"\"
   # Name: [Descriptive Name]
   # Description: [Mathematical justification of why this filter works]
   \"\"\"

[Output Requirement]
Output ONLY the Python code block containing the single requested function. Do not output test code or the main search algorithm.
"""

def evaluate_filter_logic(code_string, ground_truth_data):
    """
    Evaluates the performance of an AI-generated filter (pruning heuristic).
    ground_truth_data format: { p: {'A': valid_A, 'x0': valid_x0} }
    """
    try:
        # 1. Dynamically load the AI-generated filter into a safe namespace
        namespace = {}
        exec(code_string, namespace)

        ai_filter_func = None
        for name, func in namespace.items():
            if name.startswith("filter_") and callable(func):
                ai_filter_func = func
                break

        if not ai_filter_func:
            return False, "❌ Missing Function: Could not find a valid function definition starting with `filter_`."

        # 2. Core evaluation loop
        total_pruning_rate = 0

        for p, truth in ground_truth_data.items():
            true_A = truth['A']
            true_x0 = truth['x0']

            # --- Metric A: Absolute Safety Check (0% False Negative) ---
            # If the filter discards the known valid answer, the strategy is immediately rejected.
            if not ai_filter_func(true_A, true_x0, p):
                return False, f"❌ FATAL ERROR (False Negative): For p={p}, the known valid A={true_A} was incorrectly discarded by the filter."

            # --- Metric B: Pruning Efficiency Test (Pruning Rate) ---
            # Count how many guaranteed failures (invalid A values) the filter successfully removes.
            discarded_count = 0
            total_tested = p - 1

            for test_A in range(1, p):
                if test_A == true_A:
                    continue  # Skip the ground truth value

                # If the filter returns False, it successfully pruned an invalid candidate.
                if not ai_filter_func(test_A, None, p):
                    discarded_count += 1

            pruning_rate = discarded_count / total_tested
            total_pruning_rate += pruning_rate

        # 3. Comprehensive Scoring
        avg_pruning = (total_pruning_rate / len(ground_truth_data)) * 100

        if avg_pruning == 0:
            return False, "⚠️ Ineffective Strategy: The filter is safe but did not prune any candidates (0.00% reduction rate)."

        return True, f"✅ SUCCESS: 0% False Negatives. Average search space pruned: {avg_pruning:.2f}%."

    except Exception as e:
        return False, f"❌ Execution Crash: {str(e)}"


def run_optimization_loop(iterations=3):
    TEST_GROUND_TRUTH = {
        101: {'A': 4, 'x0': 53},
        103: {'A': 1, 'x0': 62},

        10007: {'A': 2, 'x0': 123},
        10009: {'A': 5, 'x0': 456},

        100003: {'A': 10, 'x0': 789},
        100019: {'A': 7, 'x0': 321}
    }

    # 1. Initial User Prompt
    current_prompt = """Our current approach relies on randomly guessing `A` and `x0` and running the full `pp_verify` loop, which is computationally expensive. 

To help you find structural patterns, here are the valid Pomerance triples for the first few primes over 100:
p=101 -> (101, 4, 53)
p=103 -> (103, 1, 62)
p=107 -> (107, 1, 35)
p=109 -> (109, 3, 51)
p=113 -> (113, 2, 85)

Please analyze this and write an optimized filtering strategy."""

    for i in range(iterations):
        print(f"\n--- 🚀 Agent Iteration Round {i + 1} ---")

        current_next_id = get_next_filter_id(STRATEGY_FILE)

        dynamic_system_prompt = get_system_prompt(current_next_id)

        messages = [
            {"role": "system", "content": dynamic_system_prompt},
            {"role": "user", "content": current_prompt}
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2
        )

        ai_response = response.choices[0].message.content

        code_match = re.search(r"```python(.*?)```", ai_response, re.DOTALL)
        if not code_match:
            print("⚠️ AI did not output a standard code block. Trying next round.")
            current_prompt = "You did not provide the code enclosed in ```python ... ``` blocks. Please output the code properly."
            continue

        generated_code = code_match.group(1).strip()
        print("💡 AI generated a new algorithm. Running local tests...")

        success, result = evaluate_filter_logic(generated_code, TEST_GROUND_TRUTH)

        if success:
            print(result)

            with open(STRATEGY_FILE, "a", encoding="utf-8") as f:
                f.write(f"\n\n# ==========================================\n")
                f.write(f"# Generated at Iteration: {i + 1} | Filter ID: {current_next_id:02d}\n")
                f.write(f"# ==========================================\n")
                f.write(generated_code + "\n")
            print(f"💾 Appended Filter {current_next_id:02d} to {STRATEGY_FILE}")

            desc_match = re.search(r"# Description:\s*(.*)", generated_code)
            strategy_desc = desc_match.group(1).strip() if desc_match else "Discovered by AI in runtime optimization."

            ledger_path = "../output/ledger.json"
            ledger_data = []
            if os.path.exists(ledger_path):
                with open(ledger_path, "r", encoding="utf-8") as lf:
                    try:
                        ledger_data = json.load(lf)
                    except:
                        pass

            new_record = {
                "id": f"{current_next_id:02d}",
                "name": f"filter_{current_next_id:02d}_algorithm",
                "description": strategy_desc
            }
            ledger_data.append(new_record)

            with open(ledger_path, "w", encoding="utf-8") as lf:
                json.dump(ledger_data, lf, indent=2, ensure_ascii=False)
            print(f"🧠 Updated long-term memory ledger: {ledger_path}")
        else:
            print(f"⚠️ Test Failed! Reason:\n{result}")
            current_prompt = f"The generated code failed with the following error during evaluation:\n{result}\nPlease fix this mathematical or logical bug and output the corrected complete code."

if __name__ == "__main__":
    run_optimization_loop(iterations=3)