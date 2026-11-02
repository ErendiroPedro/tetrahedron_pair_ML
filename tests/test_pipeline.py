import json
import math

import pytest
import yaml

from src.pipeline import PipelineOrchestrator
from src.training import trainer as trainer_module
from tests.conftest import TASK


def write_config(config, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


def use_processed_data_in(config, folder):
    config["processor_config"]["dataset_paths"]["processed_data"] = str(folder)
    config["model_config"]["processed_data_path"] = str(folder)
    config["trainer_config"]["processed_data_path"] = str(folder)


def only_run_folder(tmp_path):
    (run_folder,) = (tmp_path / "artifacts").iterdir()
    return run_folder


def strict_json(path):
    def reject(constant):
        raise ValueError(f"{path.name} contains {constant}, which is not valid JSON")
    return json.loads(path.read_text(), parse_constant=reject)


def test_a_full_run_processes_trains_evaluates_and_saves_everything(config, tmp_path):
    use_processed_data_in(config, tmp_path / "processed")

    PipelineOrchestrator(write_config(config, tmp_path)).run()

    run_folder = only_run_folder(tmp_path)
    assert sorted(path.name for path in run_folder.iterdir()) == ["config.yaml", "evaluation_report.json", "loss_plot.png", "model"]
    assert sorted(path.name for path in (run_folder / "model").iterdir()) == ["model.pt", "state_dict.pth", "summary.txt"]
    assert (tmp_path / "processed" / "train" / "train_data.csv").exists()
    report = strict_json(run_folder / "evaluation_report.json")
    assert report["task_type"] == TASK and math.isfinite(report["training_metrics"]["final_val_loss"])


def test_a_saved_model_can_be_reloaded_and_evaluates_identically(config, tmp_path):
    config["processor_config"]["skip_processing"] = True  # reuse the session's processed data
    PipelineOrchestrator(write_config(config, tmp_path)).run()
    first_run = only_run_folder(tmp_path)

    config["trainer_config"]["skip_training"] = True
    config["model_config"]["skip_building"] = True
    config["evaluator_config"]["model_path"] = str(first_run / "model" / "model.pt")
    config["artifacts_config"]["save_artifacts_to"] = str(tmp_path / "second_run")
    PipelineOrchestrator(write_config(config, tmp_path)).run()

    first, second = (strict_json(folder / "evaluation_report.json") for folder in (first_run, next((tmp_path / "second_run").iterdir())))
    for key in ("overall_classification_metrics", "overall_regression_metrics"):
        assert second[key] == pytest.approx(first[key], rel=1e-6)
    for name, dataset_report in first["dataset_reports"].items():
        assert second["dataset_reports"][name]["classification_performance"] == dataset_report["classification_performance"]
        assert "training_metrics" not in second, "a loaded model has no training metrics"


def test_a_non_finite_loss_fails_the_run_and_records_the_error(config, tmp_path, monkeypatch):
    config["processor_config"]["skip_processing"] = True
    monkeypatch.setitem(trainer_module.LOSS_BY_TASK, TASK, lambda predictions, targets: predictions.sum() * float("nan"))
    orchestrator = PipelineOrchestrator(write_config(config, tmp_path))

    with pytest.raises(FloatingPointError, match="not finite"):
        orchestrator.run()

    run_folder = only_run_folder(tmp_path)
    error = json.loads((run_folder / "error.json").read_text())
    assert error["step"] == "model_acquisition" and "not finite" in error["error"] and "Traceback" in error["traceback"]
    assert not (run_folder / "evaluation_report.json").exists()


def test_an_invalid_config_fails_before_anything_is_created(config, tmp_path):
    config["trainer_config"]["epochs"] = 0

    with pytest.raises(ValueError, match="trainer_config.epochs"):
        PipelineOrchestrator(write_config(config, tmp_path))

    assert not (tmp_path / "artifacts").exists()
