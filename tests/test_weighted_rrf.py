
import pytest

from documind.hybrid import reciprocal_rank_fusion


def test_semantic_weight_favors_semantic_ranking():
    semantic = [
        {"id": "A", "source": "a.txt"},
        {"id": "B", "source": "b.txt"},
    ]
    keyword = [
        {"id": "B", "source": "b.txt"},
        {"id": "A", "source": "a.txt"},
    ]

    results = reciprocal_rank_fusion(
        semantic,
        keyword,
        semantic_weight=0.7,
        keyword_weight=0.3,
    )

    assert results[0]["id"] == "A"


def test_keyword_weight_favors_keyword_ranking():
    semantic = [
        {"id": "A"},
        {"id": "B"},
    ]
    keyword = [
        {"id": "B"},
        {"id": "A"},
    ]

    results = reciprocal_rank_fusion(
        semantic,
        keyword,
        semantic_weight=0.3,
        keyword_weight=0.7,
    )

    assert results[0]["id"] == "B"


def test_zero_keyword_weight():
    semantic = [{"id": "A"}]
    keyword = [{"id": "B"}]

    results = reciprocal_rank_fusion(
        semantic,
        keyword,
        semantic_weight=1.0,
        keyword_weight=0.0,
    )

    assert [r["id"] for r in results] == ["A"]


@pytest.mark.parametrize(
    "semantic_weight,keyword_weight",
    [
        (-0.1, 1.1),
        (1.1, -0.1),
        (0.0, 0.0),
    ],
)
def test_invalid_weights(semantic_weight, keyword_weight):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            [{"id": "A"}],
            [{"id": "B"}],
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
        )
