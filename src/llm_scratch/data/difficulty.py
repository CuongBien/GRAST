"""Seeded arithmetic trees with independently configurable size, depth and prompt length."""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from llm_scratch.utils.config import config_digest


@dataclass(frozen=True)
class Expression:
    value: int | None = None
    operator: str | None = None
    left: Expression | None = None
    right: Expression | None = None

    @property
    def operations(self) -> int:
        return 0 if self.operator is None else 1 + self.left.operations + self.right.operations

    @property
    def depth(self) -> int:
        return 0 if self.operator is None else 1 + max(self.left.depth, self.right.depth)

    def evaluate(self) -> Fraction:
        if self.operator is None:
            return Fraction(self.value)
        left, right = self.left.evaluate(), self.right.evaluate()
        if self.operator == "+":
            return left + right
        if self.operator == "-":
            return left - right
        if self.operator == "*":
            return left * right
        if self.operator == "/":
            return left / right
        raise ValueError(f"Unknown operator: {self.operator}")

    def as_dict(self) -> dict[str, Any]:
        if self.operator is None:
            return {"value": self.value}
        return {
            "operator": self.operator,
            "left": self.left.as_dict(),
            "right": self.right.as_dict(),
        }

    def canonical(self) -> Any:
        if self.operator is None:
            return self.value
        if self.operator in ("+", "*"):
            terms = []

            def gather(node):
                if node.operator == self.operator:
                    gather(node.left)
                    gather(node.right)
                else:
                    terms.append(node.canonical())

            gather(self)
            return [self.operator, sorted(terms, key=config_digest)]
        return [self.operator, self.left.canonical(), self.right.canonical()]


def expression_from_dict(value: dict) -> Expression:
    if set(value) == {"value"} and type(value["value"]) is int:
        return Expression(value=value["value"])
    if set(value) != {"operator", "left", "right"} or value["operator"] not in ("+", "-", "*", "/"):
        raise ValueError("Invalid expression tree")
    return Expression(
        operator=value["operator"],
        left=expression_from_dict(value["left"]),
        right=expression_from_dict(value["right"]),
    )


def build_expression(
    rng: random.Random,
    operations: int,
    depth: int,
    minimum: int,
    maximum: int,
    operators: list[str],
) -> Expression:
    if operations == 0 and depth == 0:
        return Expression(value=rng.randint(minimum, maximum))
    if not 1 <= depth <= operations <= 2**depth - 1:
        raise ValueError("Binary-tree depth must satisfy depth <= operations <= 2**depth - 1")
    cap = 2 ** (depth - 1) - 1
    deep_count = rng.randint(max(depth - 1, operations - 1 - cap), min(cap, operations - 1))
    other_count = operations - 1 - deep_count
    other_depth = (
        rng.randint(other_count.bit_length(), min(other_count, depth - 1)) if other_count else 0
    )
    children = [
        build_expression(rng, deep_count, depth - 1, minimum, maximum, operators),
        build_expression(rng, other_count, other_depth, minimum, maximum, operators),
    ]
    rng.shuffle(children)
    return Expression(operator=rng.choice(operators), left=children[0], right=children[1])


def render_question(expression: Expression, length: str) -> str:
    """Length changes wording only; dependency depth is the reasoning-step proxy."""
    statements = []
    descriptions = {"+": "sum of", "-": "difference of", "*": "product of", "/": "quotient of"}

    def render(node):
        if node.operator is None:
            return str(node.value)
        left, right = render(node.left), render(node.right)
        name = f"R{len(statements) + 1}"
        statements.append(f"{name} is the {descriptions[node.operator]} {left} and {right}.")
        return name

    final = render(expression)
    core = " ".join(statements) + f" What is {final}? Give only the numeric answer."
    padding = {
        "short": "",
        "medium": "A student writes quantities on a worksheet. Follow the definitions in order. ",
        "long": (
            "During a quiet study session, a student organizes a worksheet "
            "about abstract quantities. The page also describes a sunny afternoon "
            "and a blue notebook; these details do not change "
            "any value. Only the definitions below are relevant to the requested calculation. "
            "Read each definition carefully and keep exact values until the final answer. "
        ),
    }
    if length not in padding:
        raise ValueError("prompt_length must be short, medium or long")
    return padding[length] + core


def generate_math(levels: list[dict], per_level: int, seed: int) -> list[dict]:
    rows, seen = [], set()
    for level in levels:
        rng = random.Random(int(config_digest([seed, "math-v1", level["name"]]), 16))
        accepted, attempts = 0, 0
        while accepted < per_level:
            attempts += 1
            if attempts > max(10000, per_level * 1000):
                raise ValueError(f"Insufficient unique math questions for level {level['name']}")
            tree = build_expression(
                rng,
                level["operations"],
                level["reasoning_steps"],
                level["operand_min"],
                level["operand_max"],
                level["operators"],
            )
            try:
                answer = tree.evaluate()
            except ZeroDivisionError:
                continue
            if abs(answer.numerator) > 10**9 or answer.denominator > 10**9:
                continue
            semantic_id = config_digest(tree.canonical())
            if semantic_id in seen:
                continue
            seen.add(semantic_id)
            question = render_question(tree, level["prompt_length"])
            rows.append(
                {
                    "question_id": f"math-{semantic_id}",
                    "source": "grast-arithmetic-v1",
                    "difficulty": level["name"],
                    "question": question,
                    "answer": str(answer),
                    "canonical_answer": str(answer),
                    "operation_count": tree.operations,
                    "reasoning_steps": tree.depth,
                    "prompt_length": level["prompt_length"],
                    "prompt_word_count": len(question.split()),
                    "semantic_id": semantic_id,
                    "expression": tree.as_dict(),
                }
            )
            accepted += 1
    return rows
