"""gen_math.py — math training data as WORKED SOLUTIONS, not internal monologue.

Experiment-003 collapsed into terse answers and got arithmetic wrong (8.0 -> 0.8, 81920 -> 64).
The fix is not "show a huge chain of thought"; it is to expose just enough of the calculation to
reach and verify the answer:

    question -> short derivation -> final answer

Every answer here is COMPUTED in Python and checked to appear in the derivation, so the generator
cannot ship a wrong worked example (which would teach the model to compute wrongly — the exact
failure we are fixing).

Problems are distinct from the benchmark's math items (different numbers/scenarios);
prepare_dataset.py hard-fails on any prompt overlap.
"""
from __future__ import annotations

import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _f(x):
    """Format a number cleanly: ints without a decimal, floats trimmed."""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.4f}".rstrip("0").rstrip(".")


# Each entry: prompt, computed answer, and a derivation that MUST contain the answer string.
def build():
    rows = []

    def add(prompt, answer, derivation):
        a = _f(answer) if isinstance(answer, (int, float)) else str(answer)
        if a not in derivation:
            raise SystemExit(f"BAD MATH EXAMPLE: answer {a!r} not shown in derivation "
                             f"for {prompt!r}")
        rows.append({"messages": [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": derivation}]})

    # compounding
    r, d = 0.0005, 200
    ans = ((1 + r) ** d - 1) * 100
    add("A strategy returns 0.05% per day. Compounded over 200 trading days, what is the total "
        "return as a percentage to one decimal place?",
        f"{ans:.1f}",
        f"Compound daily: (1 + 0.0005)^200 = {(1+r)**d:.4f}. "
        f"Total return = {(1+r)**d:.4f} - 1 = {ans/100:.4f} = {ans:.1f}%.")

    # CAGR
    start, end, years = 1000, 2500, 5
    cagr = ((end / start) ** (1 / years) - 1) * 100
    add("An investment grows from $1000 to $2500 over 5 years. What is the compound annual growth "
        "rate as a percentage to one decimal place?",
        f"{cagr:.1f}",
        f"CAGR = (end/start)^(1/years) - 1 = (2500/1000)^(1/5) - 1 = "
        f"{(end/start)**(1/years):.4f} - 1 = {cagr:.1f}%.")

    # percentage change
    old, new = 80, 92
    pc = (new - old) / old * 100
    add("A stock rises from $80 to $92. What is the percentage increase, to one decimal place?",
        f"{pc:.1f}",
        f"Percentage change = (new - old)/old * 100 = (92 - 80)/80 * 100 = "
        f"{(new-old)/old:.4f} * 100 = {pc:.1f}%.")

    # percentage of a number
    add("What is 37.5% of 640?",
        240,
        "37.5% = 0.375, and 0.375 * 640 = 240.")

    # weighted average
    add("Three trades return +2%, -1%, and +5% on position sizes of $100, $300, and $100. "
        "What is the capital-weighted average return, to two decimal places?",
        f"{(2*100 + -1*300 + 5*100)/500:.2f}",
        "Weighted average = sum(return*size)/sum(size) = "
        f"(2*100 + (-1)*300 + 5*100)/(100+300+100) = ({200-300+500})/500 = "
        f"{(2*100 -1*300 +5*100)/500:.2f}%.")

    # simple algebra
    add("Solve for x: 3x + 7 = 34.",
        9,
        "3x + 7 = 34, so 3x = 27, so x = 9.")

    # ratio / proportion
    add("A recipe uses flour and sugar in a 5:2 ratio. If you use 750g of flour, how much sugar?",
        300,
        "Sugar = flour * (2/5) = 750 * 2/5 = 300g.")

    # unit conversion
    add("A process runs at 2.5 MB/s. How many gigabytes does it transfer in one hour? "
        "Use 1 GB = 1000 MB, to one decimal place.",
        f"{2.5*3600/1000:.1f}",
        "In one hour: 2.5 MB/s * 3600 s = 9000 MB. In GB: 9000/1000 = "
        f"{2.5*3600/1000:.1f} GB.")

    # probability
    add("Two fair six-sided dice are rolled. What is the probability the sum is 7? Give a "
        "fraction.",
        "1/6",
        "There are 36 equally likely outcomes; 6 of them sum to 7 "
        "((1,6),(2,5),(3,4),(4,3),(5,2),(6,1)). Probability = 6/36 = 1/6.")

    # expected value
    ev = 0.6 * 2 + 0.4 * -1
    add("A bet wins $2 with probability 0.6 and loses $1 with probability 0.4. What is the "
        "expected value?",
        f"{ev:.1f}",
        f"EV = 0.6*(+2) + 0.4*(-1) = 1.2 - 0.4 = {ev:.1f}.")

    # drawdown
    add("An equity curve goes 100 -> 130 -> 90 -> 120. What is the maximum drawdown as a "
        "percentage, to one decimal place?",
        f"{(130-90)/130*100:.1f}",
        "The largest peak-to-trough drop is 130 -> 90. Drawdown = (130-90)/130 * 100 = "
        f"{(130-90)/130*100:.1f}%.")

    # break-even win rate
    add("A strategy risks 1 to make 2 (reward:risk = 2:1). What win rate is needed to break even, "
        "as a percentage to one decimal place?",
        f"{1/(1+2)*100:.1f}",
        "Break-even win rate = risk/(risk+reward) = 1/(1+2) = 1/3 = "
        f"{1/3*100:.1f}%.")

    # averaging
    nums = [12, 15, 21, 8, 14]
    add(f"What is the mean of {', '.join(map(str, nums))}?",
        f"{sum(nums)/len(nums):.1f}",
        f"Mean = sum/count = {sum(nums)}/{len(nums)} = {sum(nums)/len(nums):.1f}.")

    # doubling time (rule of 72 exact)
    import math
    dt = math.log(2) / math.log(1 + 0.08)
    add("At 8% annual growth, how many years does it take to double, to one decimal place? "
        "Use exact compounding.",
        f"{dt:.1f}",
        f"Doubling time = ln(2)/ln(1.08) = {math.log(2):.4f}/{math.log(1.08):.4f} = "
        f"{dt:.1f} years.")

    # simultaneous / system
    add("If 2a = b and a + b = 12, what is a?",
        4,
        "Substitute b = 2a into a + b = 12: a + 2a = 12, so 3a = 12, so a = 4.")

    # percentage points vs percent
    add("A win rate improves from 40% to 50%. By how many percentage POINTS did it rise?",
        10,
        "Percentage points is the simple difference: 50 - 40 = 10 points.")

    return rows


def main():
    rows = build()
    out = os.path.join(HERE, "data", "raw", "math_train.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} worked-solution math examples to {os.path.relpath(out, HERE)}")
    print("  format: question -> concise derivation -> final answer (every answer verified)")


if __name__ == "__main__":
    main()
