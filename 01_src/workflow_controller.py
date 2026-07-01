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
    history_context = load_history_context()

    return f"""You are an expert computational number theorist building an automated Pomerance search engine.

[Data Context]
Here is a sample dataset of primes and their valid Pomerance parameters (p, A, x0):
{data_string}

{history_context}

[Task]
Autonomously discover a statistical or algebraic constraint to prune the search space for 'A'. 
Your output must be a single Python function `apply_filter(A, x0, p)` that returns True (keep) or False (discard).

[Strict Engineering Constraints]
1. NAMING: The function MUST be named `filter_{next_id:02d}_[your_descriptive_name]`.
2. MODULARITY: Output pure filtering logic only. NO unbounded loops (to prevent infinite hanging).
3. HEURISTICS ALLOWED (NEW RULE): 
   - We now allow Heuristic filters! It is ACCEPTABLE to falsely discard some valid 'A' values (False Negatives), as long as you drastically reduce the search space.
   - However, you MUST NOT discard 100% of the valid answers. If your filter rejects everything, it will cause an infinite loop in our C-engine and be rejected.
4. DOCUMENTATION: Your function must begin with a docstring formatted exactly like this:
   \"\"\"
   # Name: [Descriptive Name]
   # Description: [Mathematical justification]
   \"\"\"

[Output Requirement]
Output ONLY the Python code block containing the single requested function.
"""

def evaluate_filter_logic(code_string, ground_truth_data):
    """
    Evaluates the funnel performance, calculating the False Negative Rate and Pruning Rate.
    """
    try:
        namespace = {}
        exec(code_string, namespace)

        ai_filter_func = None
        for name, func in namespace.items():
            if name.startswith("filter_") and callable(func):
                ai_filter_func = func
                break

        if not ai_filter_func:
            return False, "❌ Missing Function: Could not find a valid function definition."

        total_truths = len(ground_truth_data)
        missed_truths = 0
        total_pruning_rate = 0

        # Core evaluation loop
        for p, truth in ground_truth_data.items():
            true_A = truth['A']
            true_x0 = truth['x0']

            # Count False Negatives (Missed Truths)
            if not ai_filter_func(true_A, true_x0, p):
                missed_truths += 1

            # Calculate Pruning Efficiency
            discarded_count = 0
            total_tested = p - 1
            for test_A in range(1, p):
                if test_A == true_A:
                    continue
                if not ai_filter_func(test_A, None, p):
                    discarded_count += 1

            total_pruning_rate += (discarded_count / total_tested)

        # Calculate final percentages
        false_negative_rate = (missed_truths / total_truths) * 100
        avg_pruning = (total_pruning_rate / total_truths) * 100

        # Infinite Loop Prevention: If the false negative rate is 100% (all truths killed), reject!
        if false_negative_rate == 100.0:
            return False, f"❌ FATAL (Infinite Loop Risk): 100% False Negative Rate! You discarded ALL valid answers. This will cause the C-engine to hang infinitely."

        # Reject ineffective filters
        if avg_pruning == 0:
            return False, "⚠️ Ineffective Strategy: 0.00% pruning rate. The filter provides no optimization."

        # Reject if the false negative rate is too high (e.g., > 60%) and prompt for optimization
        if false_negative_rate > 60.0:
            return False, f"⚠️ High False Negative: Pruned {avg_pruning:.2f}%, but the False Negative Rate is {false_negative_rate:.2f}%. This is too aggressive. Please relax your mathematical bounds to lower the error rate!"

        # Accept as Heuristic if the false negative rate is within engineering tolerance
        if false_negative_rate > 0:
            return True, f"✅ HEURISTIC ACCEPTED: Pruned {avg_pruning:.2f}% of the search space. Current False Negative Rate is {false_negative_rate:.2f}% (acceptable engineering trade-off)."

        # Perfect pass (Deterministic)
        return True, f"✅ DETERMINISTIC SUCCESS: Perfect 0.00% False Negative Rate! Average search space pruned: {avg_pruning:.2f}%."

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

    current_prompt = """Our current approach relies on randomly guessing `A` and running the verification loop. 
    Please analyze the data and write an optimized filtering strategy. 
    Remember, heuristic algorithms are allowed, but try to keep the False Negative rate low."""

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
            print("⚠️ AI did not output a standard code block.")
            current_prompt = "You did not provide the code enclosed in ```python ... ``` blocks. Please output the code properly."
            continue

        generated_code = code_match.group(1).strip()
        print("💡 AI generated a new algorithm. Running local tests...")

        print("\n--- 📦 AI Generated Code ---")
        print(generated_code)
        print("----------------------------\n")

        success, result = evaluate_filter_logic(generated_code, TEST_GROUND_TRUTH)

        if success:
            print(result)

            annotated_code = f"# Eval Result: {result}\n{generated_code}"

            with open(STRATEGY_FILE, "a", encoding="utf-8") as f:
                f.write(f"\n\n# ==========================================\n")
                f.write(f"# Generated at Iteration: {i + 1} | Filter ID: {current_next_id:02d}\n")
                f.write(f"# ==========================================\n")
                f.write(annotated_code + "\n")
            print(f"💾 Appended Filter {current_next_id:02d} to {STRATEGY_FILE}")

            desc_match = re.search(r"# Description:\s*(.*)", generated_code)
            strategy_desc = desc_match.group(1).strip() if desc_match else "Heuristic strategy."

            ledger_path = "../output/ledger.json"
            ledger_data = []
            if os.path.exists(ledger_path):
                with open(ledger_path, "r", encoding="utf-8") as lf:
                    try:
                        ledger_data = json.load(lf)
                    except:
                        pass

            ledger_data.append({
                "id": f"{current_next_id:02d}",
                "name": f"filter_{current_next_id:02d}_algorithm",
                "description": f"{strategy_desc} [{result.split(':')[0]}]"
            })

            with open(ledger_path, "w", encoding="utf-8") as lf:
                json.dump(ledger_data, lf, indent=2, ensure_ascii=False)

            current_prompt = f"Excellent! Your previous code was accepted with the following performance: {result}\nFor the next iteration, can you create another orthogonal strategy, or optimize the logic to lower the false negative rate further?"

        else:
            print(f"⚠️ Test Failed! Reason:\n{result}")
            current_prompt = f"The generated code failed validation:\n{result}\nPlease mathematically relax your constraints to reduce the False Negative rate, ensuring you don't reject 100% of the valid answers."

if __name__ == "__main__":
    run_optimization_loop(iterations=3)