from typing import Any, Dict
from argparse import Namespace

__all__ = [
    "get_metric",
    "get_metrics"
]


def get_metric(name: str) -> Any:
    """
    Get metric function by name.

    Args:
        name (str): metric name.
    Returns:
        (Any): metric function
    """
    if name == "dice":
        from monai.metrics import DiceMetric
        return DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    elif name == "dice_coeff":
        from .metric_dicecoeff import DiceCoeffMetric
        return DiceCoeffMetric(reduction="none")
    elif name == "mae":
        from monai.metrics import MAEMetric
        return MAEMetric(reduction="none")
    elif name == "entropy":
        from .metric_entropy import EntropyMetric
        return EntropyMetric(reduction="none", act="softmax")
    elif name == "iou":
        from .metric_iou import IouMetric
        return IouMetric(reduction="none")
    else:
        print("The metric is not supported now !!!")
        return None


def get_metrics(args: Namespace) -> Dict:
    """
    Get a list of metrics by name.

    Args:
        args: metric configuration.
    Returns:
        (Dict): collection of metrics.
    """
    metric_dict = {}
    for metric in args.metrics:
        metric_dict[metric] = get_metric(metric)
    return metric_dict
