"""A tiny, explicit-state RNG for checkpointable long-running samplers.

Why not numpy/numba's usual np.random: inside a @njit function,
np.random.seed(seed) sets an *opaque*, global, un-inspectable generator
state -- there is no way to read it out mid-stream to checkpoint it, or
to hand it back in to resume exactly where a previous run left off.
Wang-Landau and multicanonical runs are long (many millions of moves)
and explicitly need to survive interruption (Phase 3b part 2), so their
RNG state has to be a plain value we can save and restore.

splitmix64 (S. Vigna / public domain reference algorithm) is used: a
single uint64 state, one multiply-xor-shift step per call, full period
2^64, good enough statistical quality for Monte Carlo (not
cryptographic). State is threaded explicitly: every call takes the
current state and returns (new_state, value) -- callers carry the
state themselves, so checkpointing it is just saving one uint64.
"""
from __future__ import annotations

import numpy as np
from numba import njit

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


@njit(cache=True, inline="always")
def next_u64(state):
    state = (state + np.uint64(0x9E3779B97F4A7C15)) & MASK64
    z = state
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & MASK64
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & MASK64
    z = z ^ (z >> np.uint64(31))
    return state, z


@njit(cache=True, inline="always")
def next_double(state):
    """Uniform double in [0, 1)."""
    state, z = next_u64(state)
    return state, (z >> np.uint64(11)) * (1.0 / 9007199254740992.0)  # 2**53


@njit(cache=True, inline="always")
def next_below(state, n):
    """Uniform int64 in [0, n). Slight modulo bias, negligible for the
    n < 128 used here (bias is O(n / 2^64))."""
    state, z = next_u64(state)
    return state, np.int64(z % np.uint64(n))


def tick(state):
    """Numba gotcha, not a style nit: a @njit function's uint64 return
    value unboxes to a plain Python int at the jit/Python boundary, and
    a plain int >= 2**63 gets re-inferred by Numba as signed int64 (not
    uint64) on the *next* call, silently corrupting the high bit via an
    arithmetic (sign-propagating) instead of logical right shift --
    this was caught by next_double's output being biased into [0, 0.5)
    instead of [0, 1) in testing, not by any type error. Call this on
    every RNG state that crosses back into pure Python between jitted
    calls (the production sweep functions avoid the problem entirely by
    keeping the whole sweep, and so all but one state hand-off, inside
    a single jitted call)."""
    return np.uint64(state)


def seed_state(seed: int) -> np.uint64:
    """A splitmix64 state is just any uint64; seed it by running the
    generator once so seed=0 and seed=1 don't produce suspiciously
    similar early outputs."""
    s = np.uint64(seed) & MASK64
    s, _ = next_u64(s)
    return s
