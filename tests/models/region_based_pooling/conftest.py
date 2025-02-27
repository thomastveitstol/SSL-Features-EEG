import pytest

from elecssl.models.region_based_pooling.region_based_pooling import RegionBasedPooling, RBPDesign, RBPPoolType


def _get_rgb_modules():
    k_1 = [2, 2, 2, 2, 2, 2, 2]
    k_2 = [3, 3, 3, 3, 3, 3, 3, 3]
    k_3 = [2, 3, 2, 3, 2, 3, 2, 3, 2]
    k_4 = [4, 3, 2, 3, 4, 3, 2, 3, 4]

    # -------------
    # Create models
    # -------------
    # Model 1
    design_0 = RBPDesign(
        cmmn_kwargs={}, num_designs=1, pooling_methods="MultiMSMean", pooling_methods_kwargs={},
        pooling_type=RBPPoolType.MULTI_CS, split_methods=["CentroidPolygons"] * 3,  # type: ignore
        split_methods_kwargs=[{"channel_positions": ["LEMON", "Wang"], "k": [3, 3, 3, 3, 3, 3, 3, 3],  # type: ignore
                               "min_nodes": 6},
                              {"channel_positions": ["LEMON", "Wang"], "k": [2, 3, 2, 3, 2, 3, 2, 3, 2],
                               "min_nodes": 4},
                              {"channel_positions": ["LEMON", "Wang"], "k": [4, 3, 2, 3, 4, 3, 2, 3, 4],
                               "min_nodes": 2}]
    )
    model_1 = RegionBasedPooling((design_0,))

    # Model 2
    design_0 = RBPDesign(
        cmmn_kwargs={}, num_designs=1, pooling_methods="MultiMSSharedRocketHeadRegion",
        pooling_methods_kwargs={"bias": False, "latent_search_features": 45, "max_receptive_field": 26,
                                "num_kernels": 13, "share_search_receiver_modules": True},
        pooling_type=RBPPoolType.MULTI_CS, split_methods=["CentroidPolygons"],  # type: ignore
        split_methods_kwargs=[{"channel_positions": ["LEMON", "Wang", "DortmundVital"],  # type: ignore
                               "k": [3, 3, 3, 3, 3, 3, 3, 3],
                               "min_nodes": 1}]
    )
    design_1 = RBPDesign(
        cmmn_kwargs={}, num_designs=1, pooling_methods="MultiMSMean", pooling_methods_kwargs={},
        pooling_type=RBPPoolType.MULTI_CS, split_methods=["CentroidPolygons"],  # type: ignore
        split_methods_kwargs=[{"channel_positions": ["LEMON", "Wang", "DortmundVital"],  # type: ignore
                               "k": [3, 3, 3, 3, 3, 3, 3, 3],
                               "min_nodes": 2}]
    )
    design_2 = RBPDesign(
        cmmn_kwargs={}, num_designs=1, pooling_methods="MultiMSSharedRocketHeadRegion",
        pooling_methods_kwargs={"bias": False, "latent_search_features": 8, "max_receptive_field": 27,
                                "num_kernels": 46, "share_search_receiver_modules": True},
        pooling_type=RBPPoolType.MULTI_CS, split_methods=["CentroidPolygons"],  # type: ignore
        split_methods_kwargs=[{"channel_positions": ["LEMON", "Wang", "DortmundVital"],  # type: ignore
                               "k": [4, 3, 2, 3, 4, 3, 2, 3, 4],
                               "min_nodes": 5}]
    )
    model_2 = RegionBasedPooling((design_0, design_1, design_2))

    # Model 3
    design_0 = RBPDesign(
        cmmn_kwargs={}, num_designs=1, pooling_methods="MultiMSMean", pooling_methods_kwargs={},
        pooling_type=RBPPoolType.MULTI_CS, split_methods=["CentroidPolygons"] * 4,  # type: ignore
        split_methods_kwargs=[{"channel_positions": ["LEMON", "Wang", "DortmundVital"],  # type: ignore
                               "k": k_4, "min_nodes": 4},
                              {"channel_positions": ["LEMON", "Wang", "DortmundVital"],
                               "k": k_2, "min_nodes": 4},
                              {"channel_positions": ["LEMON", "Wang", "DortmundVital"],
                               "k": k_1, "min_nodes": 6},
                              {"channel_positions": ["LEMON", "Wang", "DortmundVital"],
                               "k": k_1, "min_nodes": 1}
                              ]
    )
    model_3 = RegionBasedPooling((design_0,))

    # Model 4
    design_0 = RBPDesign(
        cmmn_kwargs={}, num_designs=1, pooling_methods="MultiMSMean", pooling_methods_kwargs={},
        pooling_type=RBPPoolType.MULTI_CS, split_methods=["CentroidPolygons"] * 3,  # type: ignore
        split_methods_kwargs=[{"channel_positions": ["LEMON", "Wang", "DortmundVital"],  # type: ignore
                               "k": k_2, "min_nodes": 1},
                              {"channel_positions": ["LEMON", "Wang", "DortmundVital"],
                               "k": k_1, "min_nodes": 1},
                              {"channel_positions": ["LEMON", "Wang", "DortmundVital"],
                               "k": k_2, "min_nodes": 6}
                              ]
    )
    model_4 = RegionBasedPooling((design_0,))

    # Model 5
    design_0 = RBPDesign(
        cmmn_kwargs={}, num_designs=1, pooling_methods="MultiMSMean", pooling_methods_kwargs={},
        pooling_type=RBPPoolType.MULTI_CS, split_methods=["CentroidPolygons"] * 4,  # type: ignore
        split_methods_kwargs=[{"channel_positions": ["LEMON", "Wang", "DortmundVital"],  # type: ignore
                               "k": k_1, "min_nodes": 5},
                              {"channel_positions": ["LEMON", "Wang", "DortmundVital"],
                               "k": k_4, "min_nodes": 5},
                              {"channel_positions": ["LEMON", "Wang", "DortmundVital"],
                               "k": k_4, "min_nodes": 2},
                              {"channel_positions": ["LEMON", "Wang", "DortmundVital"],
                               "k": k_2, "min_nodes": 6}
                              ]
    )
    model_5 = RegionBasedPooling((design_0,))

    # Model 6
    design_0 = RBPDesign(
        cmmn_kwargs={}, num_designs=1, pooling_methods="MultiMSMean", pooling_methods_kwargs={},
        pooling_type=RBPPoolType.MULTI_CS, split_methods=["CentroidPolygons"],  # type: ignore
        split_methods_kwargs=[{"channel_positions": ["LEMON", "Wang", "DortmundVital"],  # type: ignore
                               "k": k_4, "min_nodes": 3}
                              ]
    )
    design_1 = RBPDesign(
        cmmn_kwargs={}, num_designs=1, pooling_methods="MultiMSSharedRocket",
        pooling_methods_kwargs={"max_receptive_field": 37, "num_kernels": 19},
        pooling_type=RBPPoolType.MULTI_CS, split_methods=["CentroidPolygons"],  # type: ignore
        split_methods_kwargs=[{"channel_positions": ["LEMON", "Wang", "DortmundVital"],  # type: ignore
                               "k": k_1, "min_nodes": 3}
                              ]
    )
    design_2 = RBPDesign(
        cmmn_kwargs={}, num_designs=1, pooling_methods="MultiMSSharedRocketHeadRegion",
        pooling_methods_kwargs={"bias": False, "latent_search_features": 23, "max_receptive_field": 23,
                                "num_kernels": 34, "share_search_receiver_modules": False},
        pooling_type=RBPPoolType.MULTI_CS, split_methods=["CentroidPolygons"],  # type: ignore
        split_methods_kwargs=[{"channel_positions": ["LEMON", "Wang", "DortmundVital"],  # type: ignore
                               "k": k_3, "min_nodes": 4}
                              ]
    )
    design_3 = RBPDesign(
        cmmn_kwargs={}, num_designs=1, pooling_methods="MultiMSSharedRocket",
        pooling_methods_kwargs={"max_receptive_field": 16, "num_kernels": 70},
        pooling_type=RBPPoolType.MULTI_CS, split_methods=["CentroidPolygons"] * 2,  # type: ignore
        split_methods_kwargs=[{"channel_positions": ["LEMON", "Wang", "DortmundVital"],  # type: ignore
                               "k": k_3, "min_nodes": 1},
                              {"channel_positions": ["LEMON", "Wang", "DortmundVital"],  # type: ignore
                               "k": k_2, "min_nodes": 3}
                              ]
    )
    design_4 = RBPDesign(
        cmmn_kwargs={}, num_designs=1, pooling_methods="MultiMSSharedRocketHeadRegion",
        pooling_methods_kwargs={"bias": False, "latent_search_features": 40, "max_receptive_field": 21,
                                "num_kernels": 26, "share_search_receiver_modules": True},
        pooling_type=RBPPoolType.MULTI_CS, split_methods=["CentroidPolygons"],  # type: ignore
        split_methods_kwargs=[{"channel_positions": ["LEMON", "Wang", "DortmundVital"],  # type: ignore
                               "k": k_1, "min_nodes": 2}
                              ]
    )
    model_6 = RegionBasedPooling((design_0, design_1, design_2, design_3, design_4))

    return model_1, model_2, model_3, model_4, model_5, model_6


@pytest.fixture
def rbp_modules():
    try:
        return _get_rgb_modules()
    except OSError:
        pytest.skip("Skipping test: Fixture requires data not available on GitHub Actions")
