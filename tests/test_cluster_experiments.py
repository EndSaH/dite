import importlib.util
import json
from pathlib import Path

import numpy as np

from dite.app.config import Config
from dite.cluster.api import ClusterMetrics
from dite.flow.api import PipelineResult

_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "cluster_experiments.py"
_SPEC = importlib.util.spec_from_file_location("cluster_experiments", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
cluster_experiments = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cluster_experiments)


def _fake_pipeline_result(
    file_path: Path,
    *,
    label: int,
    name: str | None,
    initial_clusters: int,
    initial_noise: int,
    small_clusters_merged: int = 0,
    name_clusters_merged: int = 0,
    num_noise: bool = False,
) -> PipelineResult:
    metrics = ClusterMetrics(
        initial_clusters=initial_clusters,
        initial_noise=initial_noise,
        noise_repaired=0,
        small_clusters_merged=small_clusters_merged,
        name_clusters_merged=name_clusters_merged,
    )
    labels = np.array([-1 if num_noise else label], dtype=int)
    return PipelineResult(
        files=[file_path],
        contents=["alpha"],
        embeddings=np.array([[0.1, 0.2]], dtype=np.float32),
        labels=labels,
        cluster_names={} if num_noise or name is None else {label: name},
        noise_repaired=0,
        clusters_merged=small_clusters_merged + name_clusters_merged,
        cluster_metrics=metrics,
    )


def test_compare_inputs_builds_file_level_diff(monkeypatch, tmp_path: Path) -> None:
    doc = tmp_path / "doc.txt"
    doc.write_text("alpha", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "name": "sample",
                "tier": "representative",
                "description": "demo",
                "owner": "team",
                "files": [{"path": "doc.txt", "cluster_id": "demo"}],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    def fake_run_pipeline(folder, *, config, use_cache, use_knn_repair, options):
        del folder, config, use_cache, use_knn_repair
        if options.embedding_input_mode == "with_filename":
            return _fake_pipeline_result(
                doc,
                label=0,
                name="Filename Cluster",
                initial_clusters=2,
                initial_noise=0,
                small_clusters_merged=1,
            )
        return _fake_pipeline_result(
            doc,
            label=-1,
            name=None,
            initial_clusters=1,
            initial_noise=1,
            num_noise=True,
        )

    monkeypatch.setattr(cluster_experiments, "_run_pipeline", fake_run_pipeline)

    report = cluster_experiments._compare_inputs(
        tmp_path,
        config=Config(),
        use_cache=False,
        use_knn_repair=True,
    )

    assert report["runs"]["with_filename"]["summary"]["initial_num_clusters"] == 2
    assert report["runs"]["with_filename"]["process_metrics"]["num_files_total"] == 1
    assert report["runs"]["with_filename"]["constraint_metrics"] is not None
    assert report["runs"]["content_only"]["summary"]["num_noise"] == 1
    assert report["diff"]["summary"]["total_clusters_merged_delta"] == 1
    assert report["diff"]["entries"][0]["changed"] is True
    assert report["diff"]["entries"][0]["from_name"] == "Filename Cluster"
    assert report["diff"]["entries"][0]["to_noise"] is True


def test_sweep_reports_fragmentation_score(monkeypatch, tmp_path: Path) -> None:
    doc = tmp_path / "doc.txt"
    doc.write_text("alpha", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "name": "sample",
                "tier": "representative",
                "description": "demo",
                "owner": "team",
                "files": [{"path": "doc.txt", "cluster_id": "demo"}],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        cluster_experiments,
        "_sweep_configs",
        lambda config: [
            ("baseline-000", config, cluster_experiments.PipelineOptions()),
        ],
    )

    def fake_run_pipeline(folder, *, config, use_cache, use_knn_repair, options):
        del folder, config, use_cache, use_knn_repair, options
        metrics = ClusterMetrics(
            initial_clusters=3,
            initial_noise=1,
            small_cluster_merge_candidates=2,
            small_cluster_merge_skipped=1,
        )
        return PipelineResult(
            files=[doc],
            contents=["alpha"],
            embeddings=np.array([[0.1, 0.2]], dtype=np.float32),
            labels=np.array([-1], dtype=int),
            cluster_names={},
            noise_repaired=0,
            clusters_merged=0,
            cluster_metrics=metrics,
        )

    monkeypatch.setattr(cluster_experiments, "_run_pipeline", fake_run_pipeline)

    report = cluster_experiments._run_sweep(
        tmp_path,
        config=Config(),
        use_cache=False,
        use_knn_repair=True,
        extended=False,
    )

    assert report["runs"][0]["run_id"] == "baseline-000"
    assert report["runs"][0]["fragmentation_score"] == 4
    assert report["runs"][0]["summary"]["num_noise"] == 1
    assert report["runs"][0]["process_metrics"]["num_files_total"] == 1
    assert report["runs"][0]["constraint_metrics"]["cluster_id_fragmentation_total"] == 0
    partition = report["runs"][0]["partition_metrics"]
    assert partition["num_labeled_files"] == 1
    assert partition["predicted_noise_rate"] == 1.0


def test_emit_writes_json_file(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    payload = {"ok": True}
    cluster_experiments._emit(payload, output)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
