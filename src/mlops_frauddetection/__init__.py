"""Fraud-Anomoly Detection & Behavioral Analytics.

Scalable ML system for fraud & Anomoly detection and transaction behavior analysis
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mlops_frauddetection")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__author__ = "MergeDeployGraduate"
__email__ = "nshastry@depaul.edu"

__all__ = ["__version__", "__author__", "__email__"]
