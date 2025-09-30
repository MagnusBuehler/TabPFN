from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sklearn.utils import Bunch

from tabpfn.utils import fetch_dataset

# Sample dummy data
dummy_df = pd.DataFrame(
    {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [0, 1, 0]}
)
dummy_info = {
    "target_names": "target",
    "feature_names": ["feature1", "feature2"],
    "DESCR": "Dummy dataset",
}


def fetch_openml_mock(
    name: str | None = None,  # noqa: ARG001
    data_id: int | None = None,  # noqa: ARG001
    *,
    return_X_y: bool | None = None,  # noqa: ARG001
) -> Bunch:
    return Bunch(
        data=dummy_df[["feature1", "feature2"]],
        target=dummy_df["target"],
    )


@patch("sklearn.datasets.fetch_openml")
@patch("requests.get")
@patch("pandas.read_csv")
def test_fetch_huggingface_success(mock_read_csv, mock_requests_get, mock_fetch_openml):
    # Mock the Hugging Face CSV and JSON responses
    mock_read_csv.return_value = dummy_df
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.text = '{"target_names": ["target"], ' '"DESCR": "Dummy dataset"}'
    mock_requests_get.return_value = mock_response

    result = fetch_dataset(name="boston", return_X_y=False)

    assert isinstance(result, Bunch)
    assert "data" in result
    assert "target" in result
    assert result.data.shape == (3, 2)
    assert result.target.shape == (3,)
    mock_fetch_openml.assert_not_called()


@patch("tabpfn.utils.fetch_openml")
@patch("requests.get", side_effect=Exception("Simulated failure"))
@patch("pandas.read_csv", side_effect=Exception("Simulated failure"))
def test_fetch_huggingface_fallback_to_openml(
    mock_read_csv,  # noqa: ARG001
    mock_requests_get,  # noqa: ARG001
    mock_fetch_openml,
):
    # Make fetch_openml behave like your dummy function
    mock_fetch_openml.side_effect = fetch_openml_mock

    with pytest.warns(UserWarning) as record:
        result = fetch_dataset(name="boston", return_X_y=False)
    # assert UserWarning: Error loading CSV for dataset
    # assert Failed to load or parse metadata JSON for dataset

    assert isinstance(result, Bunch)
    mock_fetch_openml.assert_called_once()

    # Check warnings content
    warning_messages = [str(w.message) for w in record]
    assert any("Error loading CSV for dataset" in msg for msg in warning_messages)
    assert any(
        "Failed to load or parse metadata JSON" in msg for msg in warning_messages
    )


def test_fetch_huggingface_assertion():
    with pytest.raises(
        AssertionError,
        match="At least one of 'name' or 'data_id' must be provided.",
    ):
        fetch_dataset()


@patch("sklearn.datasets.fetch_openml")
@patch("requests.get")
@patch("pandas.read_csv")
def test_fetch_huggingface_success_target_not_in_features(
    mock_read_csv,
    mock_requests_get,
    mock_fetch_openml,  # noqa: ARG001
):
    # Mock the Hugging Face CSV and JSON responses
    mock_read_csv.return_value = dummy_df
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.text = '{"target_names": ["target"], ' '"DESCR": "Dummy dataset"}'
    mock_requests_get.return_value = mock_response

    result = fetch_dataset(name="boston", return_X_y=False)

    assert isinstance(result, Bunch)
    assert "data" in result
    assert "target" in result
    # none of the target names is allowed to be in the features!!!
    assert all(
        target_name not in result.data.columns for target_name in result.target_names
    )


@patch("sklearn.datasets.fetch_openml")
@patch("requests.get")
@patch("pandas.read_csv")
def test_fetch_huggingface_success_features_and_targets_in_frame(
    mock_read_csv,
    mock_requests_get,
    mock_fetch_openml,  # noqa: ARG001
):
    # Mock the Hugging Face CSV and JSON responses
    mock_read_csv.return_value = dummy_df
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.text = (
        '{"target_names": ["target"], "feature_names": ["feature1", "feature2"]}'
    )
    mock_requests_get.return_value = mock_response

    result = fetch_dataset(name="boston", return_X_y=False)

    assert isinstance(result, Bunch)
    assert "data" in result
    assert "target" in result
    # none of the target names is allowed to be in the features!!!
    assert all(
        target_name in result.frame.columns
        for target_name in result.target_names + result.feature_names
    )
