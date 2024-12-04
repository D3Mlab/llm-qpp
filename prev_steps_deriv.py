import os
import json
import shutil
from pathlib import Path


def read_detailed_results(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def derive_previous_steps(exp_root, exp_dir):
    original_dir = Path(exp_root) / exp_dir
    per_query_dir = original_dir / "per_query_results"

    for query_id in per_query_dir.iterdir():
        if query_id.is_dir():
            detailed_results_path = query_id / "detailed_results.json"
            if not detailed_results_path.exists():
                continue

            detailed_results = read_detailed_results(detailed_results_path)
            state_history = detailed_results.get("state_history", [])

            # Track iteration boundaries based on state history
            iteration_steps = []
            current_iteration = -1
            for index, state in enumerate(state_history):
                iteration = state.get("iteration", 0)
                if iteration != current_iteration:
                    iteration_steps.append(index)
                    current_iteration = iteration

            # Derive subdirectories for each previous iteration, excluding the final iteration
            for t in range(1, len(iteration_steps) - 1):
                derived_dir_name = f"der_T{t}_{exp_dir}"
                derived_dir = Path(exp_root) / derived_dir_name / "per_query_results" / query_id.name
                derived_dir.mkdir(parents=True, exist_ok=True)

                # Get the latest state for the current iteration
                current_state_index = iteration_steps[t] - 1 if t < len(iteration_steps) else len(state_history) - 1
                current_state = state_history[current_state_index]
                previous_history = state_history[:current_state_index + 1]
                current_state = current_state.copy()
                current_state.update({"state_history": previous_history})

                # Write derived detailed_results.json
                derived_detailed_results_path = derived_dir / "detailed_results.json"
                with open(derived_detailed_results_path, "w") as derived_file:
                    json.dump(current_state, derived_file, default=str, indent=4)

                # Write TREC file
                trec_file_path = derived_dir / "trec_results_raw.txt"
                top_k_docIDs = current_state.get("curr_top_k_docIDs", [])
                trec_results = []
                for doc_index, doc_id in enumerate(top_k_docIDs):
                    score = len(top_k_docIDs) - doc_index
                    trec_results.append(f"{query_id.name} Q0 {doc_id} {doc_index + 1} {score} my_run")

                with open(trec_file_path, "w") as trec_file:
                    trec_file.write("\n".join(trec_results))


if __name__ == "__main__":
    exp_root = "experiments/dec_2"
    exp_dir = "esci_100_ali_flash_rerank10_k10_T3_qpp"
    derive_previous_steps(exp_root, exp_dir)
