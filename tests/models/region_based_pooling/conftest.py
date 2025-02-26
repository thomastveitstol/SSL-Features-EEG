import pytest

from elecssl.models.region_based_pooling.region_based_pooling import RegionBasedPooling, RBPDesign, RBPPoolType


@pytest.fixture
def rbp_modules():
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

    return model_1, model_2
