import numpy as np
import pytest
from scipy.stats import chisquare
from src.cfr.buffer import ReservoirBuffer


def test_buffer_fills_to_capacity():
    buf = ReservoirBuffer(capacity=100)
    for i in range(100):
        buf.add(np.array([float(i)]), np.array([0.0]), 1.0)
    assert len(buf) == 100


def test_buffer_does_not_exceed_capacity():
    buf = ReservoirBuffer(capacity=100)
    for i in range(500):
        buf.add(np.array([float(i)]), np.array([0.0]), 1.0)
    assert len(buf) == 100


def test_buffer_sample_returns_correct_count():
    buf = ReservoirBuffer(capacity=1000)
    for i in range(1000):
        buf.add(np.array([float(i)]), np.array([0.0]), 1.0)
    batch = buf.sample(32)
    assert len(batch) == 32


def test_reservoir_uniform_coverage():
    """Chi-square test: reservoir sampling should cover all buckets uniformly."""
    capacity = 1000
    n_inserts = 100_000
    n_buckets = 10
    bucket_size = n_inserts // n_buckets

    buf = ReservoirBuffer(capacity=capacity)
    for i in range(n_inserts):
        buf.add(np.array([float(i)]), np.array([0.0]), 1.0)

    counts = np.zeros(n_buckets, dtype=int)
    for state, _, _ in buf.buffer:
        idx = int(state[0]) // bucket_size
        if 0 <= idx < n_buckets:
            counts[idx] += 1

    _, p_value = chisquare(counts)
    assert p_value > 0.01, (
        f"Reservoir sampling is not uniform (p={p_value:.4f}). "
        f"Bucket counts: {counts}"
    )
