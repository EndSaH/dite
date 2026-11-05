"""Display names of clusters: unique, and never cut inside a word (ASSIGN-3)."""

from __future__ import annotations

import unicodedata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import dite.cluster.api as cluster_api
import dite.cluster.name as cluster_name
from dite.app.config import Config
from dite.app.i18n import set_locale
from dite.cluster.model import ClusterMetrics, ClusterResult
from dite.cluster.name import (
    CLUSTER_NAME_OUTPUT_LIMIT,
    _display_cluster_name,
    _unique_cluster_names,
)


@pytest.mark.parametrize(
    ("name", "shown"),
    [
        ("Machine Learning Research Papers", "Machine Learning"),
        ("Taxes & Invoices & Receipts 2024", "Taxes & Invoices"),
        ("Notes on My Grandmother's Recipes", "Notes on My"),
        # A single word longer than the limit can only be cut.
        ("Supercalifragilisticexpialidocious", "Supercalifragilisticexpi"),
        ("Short Name", "Short Name"),
    ],
)
def test_display_names_are_cut_between_words(name: str, shown: str) -> None:
    assert _display_cluster_name(0, name) == shown


def test_display_names_without_spaces_are_cut_at_the_limit() -> None:
    chinese = "机器学习与深度学习相关的研究论文以及课程讲义和实验报告合集"
    assert len(chinese) > CLUSTER_NAME_OUTPUT_LIMIT

    assert _display_cluster_name(0, chinese) == chinese[:CLUSTER_NAME_OUTPUT_LIMIT]
    # An English word after Chinese text goes whole or not at all.
    assert _display_cluster_name(0, "机器学习研究论文合集 Machine Learning") == (
        "机器学习研究论文合集 Machine"
    )
    assert _display_cluster_name(0, "机器学习研究论文合集MachineLearning") == (
        "机器学习研究论文合集"
    )


def test_unique_names_ignore_case_and_unicode_normalization() -> None:
    nfd = unicodedata.normalize("NFD", "Café Notes")

    names = _unique_cluster_names(
        {3: "Finance", 1: "Café Notes", 2: "finance", 0: "Finance", 4: nfd}
    )

    assert names == {
        0: "Finance",
        1: "Café Notes",
        2: "finance_2",
        3: "Finance_3",
        4: f"{nfd}_2",
    }


def test_a_suffix_never_takes_the_name_of_another_cluster() -> None:
    names = _unique_cluster_names({0: "Topic", 1: "Topic", 2: "Topic_2"})

    assert names == {0: "Topic", 1: "Topic_3", 2: "Topic_2"}


class _Client:
    """Names each cluster after the marker word in its samples."""

    def __init__(self, names: dict[str, str]) -> None:
        self.names = names
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        content = next(name for key, name in self.names.items() if key in prompt)
        message = SimpleNamespace(content=content, reasoning_content=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(model="dummy-model", choices=[choice])


def _name_clusters(module, names: dict[str, str], *, merge: bool) -> dict[int, str]:
    set_locale("en")
    markers = list(names)
    labels = np.array([index for index in range(len(markers)) for _ in range(2)])
    result = ClusterResult(
        labels=labels,
        cluster_names={},
        repaired_mask=np.zeros(labels.shape, dtype=bool),
        metrics=ClusterMetrics(),
    )
    named = module.generate_all_cluster_names(
        client=_Client(names),
        result=result,
        contents=[f"{marker} sample {n}" for marker in markers for n in range(2)],
        files=[Path(f"{marker}-{n}.txt") for marker in markers for n in range(2)],
        config=Config(),
        embeddings=None,
        merge_same_name=merge,
        llm_model="dummy-model",
    )
    return named.cluster_names


@pytest.mark.parametrize("module", [cluster_api, cluster_name], ids=["api", "name"])
def test_naming_gives_every_cluster_a_name_of_its_own(module) -> None:
    names = _name_clusters(
        module,
        {
            "alpha": "Machine Learning Research Papers",
            "bravo": "machine learning research notes",
            "charlie": "Invoices",
        },
        merge=False,
    )

    assert names == {
        0: "Machine Learning",
        1: "machine learning_2",
        2: "Invoices",
    }


@pytest.mark.parametrize("module", [cluster_api, cluster_name], ids=["api", "name"])
def test_merging_same_names_still_merges_before_names_are_made_unique(
    module,
) -> None:
    names = _name_clusters(
        module,
        {"alpha": "Invoices", "bravo": "Invoices", "charlie": "INVOICES"},
        merge=True,
    )

    assert names == {0: "Invoices", 2: "INVOICES_2"}
