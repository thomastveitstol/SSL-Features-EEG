import numpy
import pandas
import seaborn
from matplotlib import pyplot

from elecssl.data.paths import get_ai_mind_cantab_and_sociodemographic_ai_dev_path


def main():
    # Load data
    path = get_ai_mind_cantab_and_sociodemographic_ai_dev_path() / "ai-mind_scd-clinical-data_ai-dev_2025-03-12.csv"
    df = pandas.read_csv(path, usecols=["participant_id", "age", "ptau217", "site_name"])
    df["log_ptau217"] = numpy.log(df["ptau217"])

    # -----------
    # Plotting
    # -----------
    for site in sorted(set(df["site_name"])):
        seaborn.regplot(df[df["site_name"] == site], x="age", y="log_ptau217", label=site, x_jitter=False)

    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    main()
