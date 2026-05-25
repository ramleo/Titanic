"""
tests/test_pipeline.py
Titanic ML Pipeline — Full Test Suite
Generated: 2026-05-25
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers — resolve paths relative to project root regardless of CWD
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

def model_path(filename: str) -> str:
    return os.path.join(MODELS_DIR, filename)


# ---------------------------------------------------------------------------
# Shared sample passengers
# ---------------------------------------------------------------------------
FEMALE_1ST = {
    "Pclass": 1, "Sex": "female", "Age": 29.0,
    "SibSp": 0, "Parch": 0, "Fare": 100.0,
    "Embarked": "S", "Title": "Mrs",
}

MALE_3RD = {
    "Pclass": 3, "Sex": "male", "Age": 22.0,
    "SibSp": 1, "Parch": 0, "Fare": 7.25,
    "Embarked": "S", "Title": "Mr",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def pipeline():
    import joblib
    return joblib.load(model_path("final_pipeline.pkl"))


@pytest.fixture(scope="session")
def preprocessor():
    import joblib
    return joblib.load(model_path("preprocessor.pkl"))


@pytest.fixture(scope="session")
def label_encoder():
    import joblib
    return joblib.load(model_path("label_encoder.pkl"))


@pytest.fixture(scope="session")
def test_data():
    import joblib
    X_test_raw = joblib.load(model_path("X_test_raw.pkl"))
    y_test = np.load(model_path("y_test.npy"))
    return X_test_raw, y_test


# ---------------------------------------------------------------------------
# Test 1 — Artifact Integrity
# ---------------------------------------------------------------------------
class TestArtifactIntegrity:
    """All required model files exist and load without error."""

    REQUIRED_FILES = [
        "final_pipeline.pkl",
        "preprocessor.pkl",
        "label_encoder.pkl",
        "X_test_raw.pkl",
        "y_test.npy",
        "X_train_raw.pkl",
        "y_train.npy",
    ]

    @pytest.mark.parametrize("filename", REQUIRED_FILES)
    def test_file_exists(self, filename):
        path = model_path(filename)
        assert os.path.isfile(path), f"Missing artifact: {path}"

    def test_final_pipeline_loads(self, pipeline):
        assert pipeline is not None

    def test_preprocessor_loads(self, preprocessor):
        assert preprocessor is not None

    def test_label_encoder_loads(self, label_encoder):
        assert label_encoder is not None

    def test_X_test_raw_loads(self, test_data):
        X, _ = test_data
        assert isinstance(X, pd.DataFrame)
        assert len(X) > 0

    def test_y_test_loads(self, test_data):
        _, y = test_data
        assert isinstance(y, np.ndarray)
        assert len(y) > 0


# ---------------------------------------------------------------------------
# Test 2 — Pipeline Steps
# ---------------------------------------------------------------------------
class TestPipelineSteps:
    """Pipeline contains required named steps."""

    def test_has_preprocessor_step(self, pipeline):
        assert "preprocessor" in pipeline.named_steps, \
            "Pipeline missing 'preprocessor' step"

    def test_has_model_step(self, pipeline):
        assert "model" in pipeline.named_steps, \
            "Pipeline missing 'model' step"

    def test_step_count(self, pipeline):
        assert len(pipeline.steps) >= 2, \
            f"Expected at least 2 steps, got {len(pipeline.steps)}"


# ---------------------------------------------------------------------------
# Test 3 — Single sample: 1st-class female → survived (label=1)
# ---------------------------------------------------------------------------
class TestSingleSampleFemale1st:
    """Predict on a 1st-class female passenger — expect Survived=1."""

    def test_prediction_is_survived(self, pipeline):
        df = pd.DataFrame([FEMALE_1ST])
        pred = pipeline.predict(df)
        assert pred[0] == 1, \
            f"Expected label=1 (Survived) for 1st-class female, got {pred[0]}"

    def test_prediction_shape(self, pipeline):
        df = pd.DataFrame([FEMALE_1ST])
        pred = pipeline.predict(df)
        assert pred.shape == (1,)


# ---------------------------------------------------------------------------
# Test 4 — Single sample: 3rd-class male → not survived (label=0)
# ---------------------------------------------------------------------------
class TestSingleSampleMale3rd:
    """Predict on a 3rd-class male passenger — expect Survived=0."""

    def test_prediction_is_not_survived(self, pipeline):
        df = pd.DataFrame([MALE_3RD])
        pred = pipeline.predict(df)
        assert pred[0] == 0, \
            f"Expected label=0 (Not Survived) for 3rd-class male, got {pred[0]}"

    def test_prediction_shape(self, pipeline):
        df = pd.DataFrame([MALE_3RD])
        pred = pipeline.predict(df)
        assert pred.shape == (1,)


# ---------------------------------------------------------------------------
# Test 5 — Test-set accuracy >= 0.80
# ---------------------------------------------------------------------------
class TestTestSetAccuracy:
    """Overall accuracy on held-out test set."""

    def test_accuracy_threshold(self, pipeline, test_data):
        from sklearn.metrics import accuracy_score
        X_test, y_test = test_data
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n  Test-set accuracy: {acc:.4f}")
        assert acc >= 0.80, \
            f"Accuracy {acc:.4f} below required threshold of 0.80"


# ---------------------------------------------------------------------------
# Test 6 — Per-class accuracy (recall)
# ---------------------------------------------------------------------------
class TestPerClassAccuracy:
    """Class 0 recall >= 0.80, class 1 recall >= 0.65."""

    def test_class_0_recall(self, pipeline, test_data):
        from sklearn.metrics import recall_score
        X_test, y_test = test_data
        y_pred = pipeline.predict(X_test)
        recall_0 = recall_score(y_test, y_pred, pos_label=0)
        print(f"\n  Class-0 recall: {recall_0:.4f}")
        assert recall_0 >= 0.80, \
            f"Class-0 recall {recall_0:.4f} below required threshold of 0.80"

    def test_class_1_recall(self, pipeline, test_data):
        from sklearn.metrics import recall_score
        X_test, y_test = test_data
        y_pred = pipeline.predict(X_test)
        recall_1 = recall_score(y_test, y_pred, pos_label=1)
        print(f"\n  Class-1 recall: {recall_1:.4f}")
        assert recall_1 >= 0.65, \
            f"Class-1 recall {recall_1:.4f} below required threshold of 0.65"


# ---------------------------------------------------------------------------
# Test 7 — Probability output
# ---------------------------------------------------------------------------
class TestProbabilityOutput:
    """predict_proba returns valid probability matrix."""

    def test_proba_shape(self, pipeline, test_data):
        X_test, _ = test_data
        proba = pipeline.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2), \
            f"Expected shape ({len(X_test)}, 2), got {proba.shape}"

    def test_proba_in_range(self, pipeline, test_data):
        X_test, _ = test_data
        proba = pipeline.predict_proba(X_test)
        assert (proba >= 0).all() and (proba <= 1).all(), \
            "Probabilities outside [0, 1]"

    def test_proba_rows_sum_to_one(self, pipeline, test_data):
        X_test, _ = test_data
        proba = pipeline.predict_proba(X_test)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, np.ones(len(X_test)), atol=1e-6,
            err_msg="Probability rows do not sum to 1"
        )


# ---------------------------------------------------------------------------
# Test 8 — Consistency check
# ---------------------------------------------------------------------------
class TestConsistency:
    """Running predict twice on the same input yields identical results."""

    def test_predict_deterministic(self, pipeline, test_data):
        X_test, _ = test_data
        sample = X_test.head(20)
        pred1 = pipeline.predict(sample)
        pred2 = pipeline.predict(sample)
        np.testing.assert_array_equal(pred1, pred2,
            err_msg="predict() returned different results on identical input")

    def test_proba_deterministic(self, pipeline, test_data):
        X_test, _ = test_data
        sample = X_test.head(20)
        proba1 = pipeline.predict_proba(sample)
        proba2 = pipeline.predict_proba(sample)
        np.testing.assert_array_equal(proba1, proba2,
            err_msg="predict_proba() returned different results on identical input")


# ---------------------------------------------------------------------------
# Test 9 — Batch prediction
# ---------------------------------------------------------------------------
class TestBatchPrediction:
    """Predicting on 10 samples returns exactly 10 predictions."""

    def _make_batch(self) -> pd.DataFrame:
        rows = []
        for i in range(5):
            rows.append(FEMALE_1ST.copy())
            rows.append(MALE_3RD.copy())
        return pd.DataFrame(rows)

    def test_batch_predict_count(self, pipeline):
        batch = self._make_batch()
        preds = pipeline.predict(batch)
        assert len(preds) == 10, \
            f"Expected 10 predictions, got {len(preds)}"

    def test_batch_predict_proba_count(self, pipeline):
        batch = self._make_batch()
        proba = pipeline.predict_proba(batch)
        assert proba.shape[0] == 10, \
            f"Expected 10 probability rows, got {proba.shape[0]}"

    def test_batch_labels_are_binary(self, pipeline):
        batch = self._make_batch()
        preds = pipeline.predict(batch)
        assert set(preds).issubset({0, 1}), \
            f"Unexpected label values in batch: {set(preds)}"


# ---------------------------------------------------------------------------
# Test 10 — Unknown category handling (Embarked='Z')
# ---------------------------------------------------------------------------
class TestUnknownCategoryHandling:
    """Pipeline must not crash when given an unseen categorical value."""

    def test_unseen_embarked_does_not_crash(self, pipeline):
        passenger = FEMALE_1ST.copy()
        passenger["Embarked"] = "Z"   # unseen category
        df = pd.DataFrame([passenger])
        try:
            pred = pipeline.predict(df)
            assert pred.shape == (1,), \
                f"Expected 1 prediction, got shape {pred.shape}"
        except Exception as exc:
            pytest.fail(
                f"Pipeline raised an exception on unseen category 'Z': {exc}"
            )

    def test_unseen_sex_does_not_crash(self, pipeline):
        passenger = MALE_3RD.copy()
        passenger["Sex"] = "unknown"   # unseen category
        df = pd.DataFrame([passenger])
        try:
            pred = pipeline.predict(df)
            assert pred.shape == (1,)
        except Exception as exc:
            pytest.fail(
                f"Pipeline raised an exception on unseen Sex value: {exc}"
            )
