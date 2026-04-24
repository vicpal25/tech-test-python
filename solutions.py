"""
Python take-home — all five tasks in one file.
Run with: python solutions.py
"""

import re
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache


# ---------------------------------------------------------------------------
# Task 1: String & Collections
# ---------------------------------------------------------------------------

def most_common_word(text: str, stopwords: set[str] | None = None) -> str | None:
    """
    Return the most frequent word in `text`, ignoring `stopwords`.

    Approach:
    - Normalize by lowercasing and extracting only alphabetic sequences with
      re.findall — this handles punctuation and numbers without extra stripping.
    - Use Counter.most_common(1), which internally uses heapq (O(n)) rather
      than sorting the full frequency table (O(n log n)).
    - Return None when there are no valid words after filtering.
    """
    words = re.findall(r'[a-z]+', text.lower())
    if stopwords:
        words = [w for w in words if w not in stopwords]
    if not words:
        return None
    return Counter(words).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Task 2: Data Structures & Algorithms
# ---------------------------------------------------------------------------

def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """
    Merge all overlapping intervals and return the result.

    Approach:
    - Sort by start value so we only need a single forward pass.
    - Keep a running "current" interval (the last element of `merged`).
      If the next interval's start is within the current end, extend;
      otherwise append a new interval.

    Time:  O(n log n) — dominated by the sort.
    Space: O(n) — output list.
    """
    if not intervals:
        return []

    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_ivs[0][:]]  # copy first interval; don't mutate input

    for start, end in sorted_ivs[1:]:
        if start <= merged[-1][1]:
            # Overlapping — extend the current interval if needed.
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return merged


# ---------------------------------------------------------------------------
# Task 3: OOP Design & API Thinking
# ---------------------------------------------------------------------------

class Logger:
    """
    Append-only logger backed by a plain list.

    Design choices:
    - A list is ideal here: O(1) append, O(n) full scan for search.
      A more specialised structure (e.g. a trie or inverted index) would only
      pay off at a scale far beyond what this interface suggests.
    - Timestamps are stored inside the log string rather than as a separate
      field, keeping the public API simple while still making entries
      human-readable.
    - get_logs returns a shallow copy so callers can't mutate internal state.
    """

    def __init__(self) -> None:
        self._logs: list[str] = []

    def log(self, message: str) -> None:
        ts = datetime.now().isoformat(timespec='seconds')
        self._logs.append(f"[{ts}] {message}")

    def get_logs(self) -> list[str]:
        return list(self._logs)

    def search(self, query: str) -> list[str]:
        q = query.lower()
        return [entry for entry in self._logs if q in entry.lower()]


# ---------------------------------------------------------------------------
# Task 4: Debugging & Refactoring — Fibonacci
# ---------------------------------------------------------------------------

# --- Broken version (as given) ---
#
# def fib(n):
#     return fib(n - 1) + fib(n - 2)
#
# Bug: no base cases. For any n, the function recurses forever (or until
# Python raises RecursionError). fib(0) calls fib(-1) → fib(-2) → ...
# Fix: add `if n <= 1: return n` before the recursive call.

def fib_iterative(n: int) -> int:
    """
    Iterative Fibonacci — preferred for production code.

    Why iterative?
    - O(n) time, O(1) space — no call-stack overhead, no risk of hitting
      Python's recursion limit.
    - Straightforward to read once you recognise the rolling-pair pattern.

    Tradeoff vs memoized: slightly less elegant for readers who think
    recursively, but strictly better on memory.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


@lru_cache(maxsize=None)
def fib_memoized(n: int) -> int:
    """
    Memoized recursive Fibonacci.

    Why memoized?
    - Reduces naive O(2^n) exponential recursion to O(n) time / O(n) space
      by caching each sub-result exactly once.
    - The recursive structure makes the mathematical definition immediately
      legible — good for explaining the algorithm.

    Tradeoff vs iterative: O(n) call-stack depth can still hit Python's
    default limit (~1000) for large n; iterative does not.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    return fib_memoized(n - 1) + fib_memoized(n - 2)


# ---------------------------------------------------------------------------
# Task 5: Hashing & Grouping
# ---------------------------------------------------------------------------

def group_anagrams(words: list[str]) -> list[list[str]]:
    """
    Group words that are anagrams of each other.

    Key insight: two words are anagrams iff their sorted-character tuples are
    identical. Using that tuple as a dict key lets us bucket every word in a
    single O(n·k log k) pass, where k is the max word length.

    A Counter-based key would also work but is slightly heavier to construct;
    sorted tuple is idiomatic and fast enough here.
    """
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for word in words:
        key = tuple(sorted(word.lower()))
        groups[key].append(word)
    return list(groups.values())


# ---------------------------------------------------------------------------
# Smoke tests — run with: python solutions.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Task 1
    assert most_common_word("") is None
    assert most_common_word("   ") is None
    assert most_common_word("the cat sat on the mat", {"the", "on"}) in {"cat", "sat", "mat"}
    assert most_common_word("go go go") == "go"
    assert most_common_word("go go go", {"go"}) is None
    print("Task 1 ✓")

    # Task 2
    assert merge_intervals([]) == []
    assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
    assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]
    assert merge_intervals([[3, 5], [1, 2]]) == [[1, 2], [3, 5]]  # unsorted input
    print("Task 2 ✓")

    # Task 3
    logger = Logger()
    logger.log("server started")
    logger.log("user logged in")
    logger.log("server stopped")
    assert len(logger.get_logs()) == 3
    assert len(logger.search("server")) == 2
    assert len(logger.search("LOGGED")) == 1  # case-insensitive
    assert len(logger.search("missing")) == 0
    print("Task 3 ✓")

    # Task 4
    expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    assert [fib_iterative(i) for i in range(10)] == expected
    assert [fib_memoized(i) for i in range(10)] == expected
    print("Task 4 ✓")

    # Task 5
    result = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    # Sort for deterministic comparison
    result_sets = sorted([sorted(g) for g in result])
    assert result_sets == [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]
    assert group_anagrams([]) == []
    print("Task 5 ✓")

    print("\nAll tasks passed.")
