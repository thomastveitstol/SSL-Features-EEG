"""
This script was created to check the number of numpy files per pre-processed version per dataset, after the
pre-processing quit. The code gave the following output (15th of May 10:08)

AIMind: {57}
LEMON: {203}
DortmundVital: {608, 607}

-----------
15th of May 14:38
AIMind: {65, 66}
LEMON: {203}
DortmundVital: {608, 607}

-----------
17th of June:
AIMind: {646, 647}
LEMON: {203}
DortmundVital: {608, 607}

The following were not contained in all:
AIMind: {'2-328.npy'}
LEMON: set()
DortmundVital: {'sub-230.npy'}

-----------
17th of June (after re-preprocessing '2-328'):
AIMind: {647}
LEMON: {203}
DortmundVital: {608, 607}

The following were not contained in all:
AIMind: set()
LEMON: set()
DortmundVital: {'sub-230.npy'}
"""
import os

from elecssl.data.paths import get_numpy_data_storage_path


def main():
    version = "preprocessed_band_pass_ec"
    path = get_numpy_data_storage_path() / version
    datasets = ("AIMind", "LEMON", "DortmundVital")

    samples_sizes = {dataset: set() for dataset in datasets}
    not_in_all = {dataset: set() for dataset in datasets}
    for dataset in datasets:
        union = set()
        intersection = set()
        is_first = True
        for folder in os.listdir(path):
            if not os.path.isdir(path / folder):
                continue

            subjects = os.listdir(path / folder / dataset)
            union.update(subjects)
            if is_first:
                intersection.update(subjects)
                is_first = False
            else:
                intersection = intersection.intersection(subjects)

            num_subjects = len(subjects)
            samples_sizes[dataset].add(num_subjects)

        not_in_all[dataset] = union - intersection

    for dataset, samples in samples_sizes.items():
        print(f"{dataset}: {samples}")

    print("\nThe following were not contained in all:")
    for dataset, sub_ids in not_in_all.items():
        print(f"{dataset}: {sub_ids}")


if __name__ == "__main__":
    main()
