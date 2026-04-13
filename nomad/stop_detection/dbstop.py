"""Compatibility wrapper for notebook code that imports nomad.stop_detection.dbstop.

This repository does not currently ship a separate DBSTOP implementation module.
To keep notebook-style compare code executable, expose the DBSTOP API surface by
delegating to SeqScan.
"""

from __future__ import annotations

from nomad.stop_detection.density_based import seqscan, seqscan_labels


def dbstop(*args, **kwargs):
    return seqscan(*args, **kwargs)


def dbstop_labels(*args, **kwargs):
    return seqscan_labels(*args, **kwargs)

