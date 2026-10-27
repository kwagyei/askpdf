"""Stage-wise evaluation (design §11, ADR-0007): metrics, statistics and reports.

Everything here is a pure function over runs and labels, so the model-free CI gate and
the local model runs share one tested implementation.
"""
