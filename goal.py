from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class GoalConstraints:
    budget: float
    delivery_deadline_days: int
    size: Optional[str] = None
    style: Optional[str] = None
    must_have_categories: List[str] = field(default_factory=list)


@dataclass
class AgentGoal:
    """
    Defines WHAT the agent must achieve.
    This is NOT a prompt. This is a contract.
    """

    scenario: str = "professional_event_outfit"
    min_retailers: int = 3
    required_categories: List[str] = field(
        default_factory=lambda: ["Blazer", "Shirt", "Trousers", "Shoes"]
    )
    constraints: Optional[GoalConstraints] = None

    def set_constraints(
        self,
        budget: float,
        delivery_deadline_days: int,
        size: Optional[str] = None,
        style: Optional[str] = None,
        must_have_categories: Optional[List[str]] = None,
    ):
        """Capture user constraints into the goal."""
        self.constraints = GoalConstraints(
            budget=budget,
            delivery_deadline_days=delivery_deadline_days,
            size=size,
            style=style,
            must_have_categories=must_have_categories or [],
        )

    def is_fully_defined(self) -> bool:
        """Check if the agent has enough info to proceed."""
        return self.constraints is not None
