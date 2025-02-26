from matplotlib import pyplot

from elecssl.data.datasets.dortmund_vital import DortmundVital
from elecssl.data.datasets.lemon import LEMON
from elecssl.data.datasets.wang import Wang


def main():
    datasets = (DortmundVital, LEMON, Wang)

    for dataset in datasets:
        dataset().plot_2d_electrode_positions()

    pyplot.show()

if __name__ == "__main__":
    main()
