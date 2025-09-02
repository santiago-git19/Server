from .base import BasePoseDetector
from .vitpose import VitPoseDetector
from .mspn import MSPNDetector
from .hrnet import HRNetDetector
from .csp import CSPDetector
from .trt_detector import TrtDetector

__all__ = ['BasePoseDetector', 'VitPoseDetector', 'MSPNDetector', 'HRNetDetector', 'CSPDetector', 'TrtDetector']
