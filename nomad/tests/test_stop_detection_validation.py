import pandas as pd

from nomad.stop_detection.validation import AlgorithmRegistry


def labels_per_user(data, dist_thresh=30, time_thresh=120, include_border_points=True):
    return data


def test_algorithm_registry_uses_parameter_identifier_not_family_name():
    registry = AlgorithmRegistry()

    returned_family = registry.add_algorithm(
        labels_per_user,
        family="seqscan",
        dist_thresh=[30, 45],
        time_thresh=120,
        include_border_points=True,
    )

    algos = list(registry)

    assert returned_family == "seqscan"
    assert [algo["family"] for algo in algos] == ["seqscan", "seqscan"]
    assert [algo["algorithm"] for algo in algos] == [
        "001__dist_thresh-30__include_border_points-true__time_thresh-120",
        "002__dist_thresh-45__include_border_points-true__time_thresh-120",
    ]
    assert all("seqscan" not in algo["algorithm"] for algo in algos)
    assert all("labels" not in algo["algorithm"] for algo in algos)


def test_algorithm_registry_persists_identifier_in_metrics_and_timings():
    registry = AlgorithmRegistry()
    registry.add_algorithm(labels_per_user, family="seqscan", dist_thresh=30)
    algo = next(iter(registry))

    annotated = registry.annotate_metrics({"precision": 1.0}, algo)
    output = registry.time_call(algo, pd.DataFrame({"x": [1, 2, 3]}))
    timings = registry.timing_frame()

    assert output["x"].tolist() == [1, 2, 3]
    assert annotated["algorithm"] == "001__dist_thresh-30"
    assert annotated["family"] == "seqscan"
    assert timings.loc[0, "algorithm"] == "001__dist_thresh-30"
    assert timings.loc[0, "family"] == "seqscan"


def test_algorithm_registry_tracks_external_parameters_without_forwarding_them():
    registry = AlgorithmRegistry()
    registry.add_algorithm(
        labels_per_user,
        family="gridbased",
        external_params={"h3_resolution": [8, 9]},
        dist_thresh=[30, 45],
    )

    algos = list(registry)
    output = registry.time_call(algos[0], pd.DataFrame({"x": [1, 2, 3]}))
    annotated = registry.annotate_metrics({"recall": 1.0}, algos[0])

    assert [algo["algorithm"] for algo in algos] == [
        "001__dist_thresh-30__h3_resolution-8",
        "002__dist_thresh-30__h3_resolution-9",
        "003__dist_thresh-45__h3_resolution-8",
        "004__dist_thresh-45__h3_resolution-9",
    ]
    assert output["x"].tolist() == [1, 2, 3]
    assert algos[0]["params"]["h3_resolution"] == 8
    assert annotated["h3_resolution"] == 8
