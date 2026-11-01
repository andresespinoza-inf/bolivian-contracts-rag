from contracts_rag.retrieval.rrf import reciprocal_rank_fusion


def test_item_in_both_lists_beats_item_in_one():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]])
    assert fused[0][0] == "b"          # rank 2 + rank 1 beats rank 1 alone


def test_scores_follow_the_formula_with_k_60():
    fused = dict(reciprocal_rank_fusion([["a"], ["a"]], k=60))
    assert fused["a"] == 2 * (1 / 61)


def test_k_controls_how_much_top_ranks_dominate():
    low_k = dict(reciprocal_rank_fusion([["a", "b"]], k=1))
    high_k = dict(reciprocal_rank_fusion([["a", "b"]], k=1000))
    assert (low_k["a"] / low_k["b"]) > (high_k["a"] / high_k["b"])


def test_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_output_is_sorted_descending():
    scores = [s for _, s in reciprocal_rank_fusion([["a", "b", "c", "d"], ["d", "c"]])]
    assert scores == sorted(scores, reverse=True)
