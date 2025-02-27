import pytest


# -----------------
# Tests with real data
# -----------------
@pytest.mark.skip(reason="Test not implemented")
def test_real_successful_test_set_integrity():
    """Test if the test set was (1) shared for all trials and folds, and (2) not in and other train or val set, for data
    splits where this should be the case. This test uses real data"""


@pytest.mark.skip(reason="Test not implemented")
def test_real_failing_test_set_integrity():
    """Test that test set integrity fails for data splits where the test set is not always the same. This test uses real
    data"""


# -----------------
# Tests with fake data
# -----------------
@pytest.mark.skip(reason="Test not implemented")
def test_fake_successful_test_set_integrity():
    """Test if the test set was (1) shared for all trials and folds, and (2) not in and other train or val set, for data
    splits where this should be the case. This test uses fake data"""


@pytest.mark.skip(reason="Test not implemented")
def test_fake_failing_test_set_integrity():
    """Test that test set integrity fails for data splits where the test set is not always the same. This test uses fake
    data"""
