import os
import pstats
from pathlib import Path


def main():
    profile = "cprofile_20250620_062401"

    path = (Path(os.path.dirname(__file__)) / "cprofiles" / profile).with_suffix(".prof")
    p = pstats.Stats(str(path))
    p.strip_dirs().sort_stats("tottime").print_stats(100)


if __name__ == "__main__":
    main()
