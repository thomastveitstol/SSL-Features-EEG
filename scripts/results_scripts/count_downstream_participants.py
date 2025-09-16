"""
Script for counting the number of included subjects per clinical site in AI-Mind

The following was printed:
Counts per site: {'1-': 152, '2-': 66, '3-': 158, '4-': 110}
Sum: 486
"""
import pandas

from elecssl.data.paths import get_results_dir


_VALID_PREFIXES = ("1-", "2-", "3-", "4-")


def main():
    experiment_time = "2025-06-23_104856"
    experiment_name = f"experiments_{experiment_time}"

    included_subjects_path = get_results_dir() / experiment_name / "downstream_subjects" / "included.csv"

    # -------------
    # Check downstream subjects
    # -------------
    df = pandas.read_csv(included_subjects_path)

    # Sanity check
    assert (df["dataset"] == "AIMind").all()

    # Count how many rows start with each prefix
    prefix_counts = {prefix: int(df["sub_id"].str.startswith(prefix).sum()) for prefix in _VALID_PREFIXES}

    print(f"Counts per site: {prefix_counts}")
    print(f"Sum: {sum(prefix_counts.values())}")


if __name__ == "__main__":
    main()
