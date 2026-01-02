"""Detection rule entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class RuleType(str, Enum):
    """Rule types."""
    SIGNATURE = "signature"
    ANOMALY = "anomaly"
    BEHAVIORAL = "behavioral"


class RuleSeverity(str, Enum):
    """Rule severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DetectionRule:
    """Detection rule for identifying threats."""
    
    rule_id: str
    name: str
    description: str
    rule_type: RuleType
    severity: RuleSeverity
    enabled: bool = True
    pattern: Optional[str] = None
    threshold: float = 0.8
    metadata: dict = field(default_factory=dict)
    
    def matches(self, event: dict) -> bool:
        """Check if rule matches event."""
        if not self.enabled:
            return False
        
        # Simple pattern matching on event data
        if self.pattern and self.pattern in str(event):
            return True
        
        return False
    
    def get_score(self, event: dict) -> float:
        """Get threat score from 0.0 to 1.0."""
        if self.matches(event):
            return self.threshold
        return 0.0
