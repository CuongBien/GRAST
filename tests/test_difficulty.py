"""Arithmetic size, dependency depth, exact answers and wording are controlled."""

import random
from fractions import Fraction

import pytest

from llm_scratch.data.difficulty import (
    Expression,
    build_expression,
    expression_from_dict,
    generate_math,
    render_question,
)


@pytest.mark.parametrize(("operations", "depth"), [(1, 1), (3, 2), (3, 3), (5, 3), (7, 3), (7, 5)])
def test_exact_operation_count_and_dependency_depth(operations, depth):
    for seed in range(10):
        tree = build_expression(random.Random(seed), operations, depth, 1, 20, ["+", "-", "*"])
        assert tree.operations == operations
        assert tree.depth == depth
        assert expression_from_dict(tree.as_dict()) == tree


def test_exact_fraction_arithmetic():
    tree = Expression(
        operator="+",
        left=Expression(operator="/", left=Expression(value=1), right=Expression(value=3)),
        right=Expression(operator="/", left=Expression(value=1), right=Expression(value=6)),
    )
    assert tree.evaluate() == Fraction(1, 2)
    assert tree.operations == 3
    assert tree.depth == 2


def test_prompt_length_changes_no_arithmetic():
    tree = build_expression(random.Random(42), 3, 2, 1, 10, ["+"])
    prompts = [render_question(tree, length) for length in ("short", "medium", "long")]
    assert len(prompts[0].split()) < len(prompts[1].split()) < len(prompts[2].split())
    assert all(prompt.endswith(prompts[0]) for prompt in prompts)


def test_commutative_associative_variants_share_dedup_group():
    a, b, c = (Expression(value=value) for value in (2, 3, 5))
    first = Expression(operator="+", left=a, right=Expression(operator="+", left=b, right=c))
    second = Expression(operator="+", left=Expression(operator="+", left=c, right=a), right=b)
    assert first.canonical() == second.canonical()
    assert (
        Expression(operator="-", left=a, right=b).canonical()
        != Expression(operator="-", left=b, right=a).canonical()
    )


def test_seed_reproducibility_and_global_rng_isolation():
    level = {
        "name": "test",
        "operations": 3,
        "reasoning_steps": 2,
        "prompt_length": "medium",
        "operand_min": 1,
        "operand_max": 30,
        "operators": ["+", "-", "*", "/"],
    }
    before = random.getstate()
    first = generate_math([level], 30, 42)
    assert random.getstate() == before
    assert first == generate_math([level], 30, 42)
    assert first != generate_math([level], 30, 43)
    assert len({row["semantic_id"] for row in first}) == 30
    for row in first:
        assert (
            Fraction(row["canonical_answer"]) == expression_from_dict(row["expression"]).evaluate()
        )


def test_impossible_tree_and_exhausted_question_space_fail():
    with pytest.raises(ValueError, match="depth"):
        build_expression(random.Random(0), 4, 2, 1, 3, ["+"])
    level = {
        "name": "tiny",
        "operations": 1,
        "reasoning_steps": 1,
        "prompt_length": "short",
        "operand_min": 1,
        "operand_max": 1,
        "operators": ["+"],
    }
    with pytest.raises(ValueError, match="Insufficient unique"):
        generate_math([level], 2, 42)
