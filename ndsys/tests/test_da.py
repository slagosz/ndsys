import pytest
from sklearn.utils.estimator_checks import check_estimator

from ndsys.optimizers import EntropicDualAveraging


@pytest.mark.parametrize(
    "optimizers",
    [EntropicDualAveraging()]
)
def test_optimizers(optimizers):
    return check_estimator(optimizers)
