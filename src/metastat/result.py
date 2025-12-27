from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class MetaResult:
    point_est: float
    interval: list
    targeted_cov: float
    actual_cov: float
    sigma: Optional[float] = None
    tau: Optional[float] = None
    chat: Optional[float] = None
    scale: Optional[float] = None
    extra: Dict[str, float] = field(default_factory=dict)

