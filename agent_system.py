# ----------------------
# Load the modules and libraries
# ----------------------
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum, auto

# -------------------------
# load the env variables (like API keys)
# -------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API")

# -------------------------
# LLM Definition or Interface
# -------------------------

class LLMInterface:


    SYSTEM_PROMPT = """
    You extract shopping constraints from conversation.

    Return JSON ONLY.

    If a value is mentioned, extract it.
    If not mentioned, return null.

    Format:

    {
    "item_name": string,
    "budget": number or null,
    "delivery_deadline_days": number or null,
    "size": string or null,
    "style": string or null 
    }
    """

    def __init__(self, OPENAI_API_KEY: str):
        self.client = OpenAI(api_key=OPENAI_API_KEY)


    def conversational_constraint_parser(self, conversation_history: List[Dict]) -> dict:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": self.SYSTEM_PROMPT}] + conversation_history,
            temperature=0
        )

        return json.loads(response.choices[0].message.content)



# --------------------
# Goal Component
# --------------------

@dataclass
class GoalConstraints:
    item_name: Optional[str] = None
    budget: Optional[float] = None
    delivery_deadline_days: Optional[int] = None
    size: Optional[str] = None
    style: Optional[str] = None

@dataclass
class AgentGoal:
    """
    Defines WHAT the agent must achieve.
    This is NOT a prompt. This is a contract.
    """

    scenario: str = "professional_event_outfit"
    constraints: Optional[GoalConstraints] = None

    def set_constraints(
        self,
        item_name: str,
        budget: float,
        delivery_deadline_days: int,
        size: Optional[str] = None,
        style: Optional[str] = None
    ):
        """Capture user constraints into the goal."""
        self.constraints = GoalConstraints(
            item_name=item_name,
            budget=budget,
            delivery_deadline_days=delivery_deadline_days,
            size=size,
            style=style
        )

    def is_fully_defined(self) -> bool:
        if self.constraints is None:
            return False

        c = self.constraints

        return all([
            c.item_name is not None,
            c.budget is not None and c.budget > 0,
            c.delivery_deadline_days is not None and c.delivery_deadline_days > 0,
            c.size is not None,
            c.style is not None
        ])

    

    def apply_llm_constraints(self, llm_output: Dict) -> None:
        """
        Reads constraint values from LLM JSON output and applies them
        to the goal using set_constraints().

        Raises:
            ValueError: If required fields are missing or invalid.
        """

        # Validate required fields
        if "item_name" not in llm_output:
            raise ValueError("Missing required field: item_name")
        
        if "budget" not in llm_output:
            raise ValueError("Missing required field: budget")

        if "delivery_deadline_days" not in llm_output:
            raise ValueError("Missing required field: delivery_deadline_days")
    
        # Type normalization (defensive programming)
        try:
            budget = float(llm_output["budget"])
            delivery_deadline_days = int(llm_output["delivery_deadline_days"])
        except (TypeError, ValueError):
            raise ValueError("Invalid type for budget or delivery_deadline_days")

        size = llm_output.get("size")
        style = llm_output.get("style")
    
        # Apply constraints using existing contract method
        self.set_constraints(
            item_name=llm_output["item_name"],
            budget=budget,
            delivery_deadline_days=delivery_deadline_days,
            size=size,
            style=style
        )


    def update_constraints_from_llm(self, llm_output: Dict) -> None:
        """
        Incrementally update constraints from LLM output.
        """

        if self.constraints is None:
            # Initialize empty constraint object
            self.constraints = GoalConstraints(
                budget=0,
                delivery_deadline_days=0
            )        

        if llm_output.get("item_name") is not None:
            self.constraints.item_name = llm_output["item_name"]

        if llm_output.get("budget") is not None:
            self.constraints.budget = float(llm_output["budget"])

        if llm_output.get("delivery_deadline_days") is not None:
            self.constraints.delivery_deadline_days = int(llm_output["delivery_deadline_days"])

        if llm_output.get("size") is not None:
            self.constraints.size = llm_output["size"]

        if llm_output.get("style") is not None:
            self.constraints.style = llm_output["style"]



# ----------------------
# State Component
# ----------------------

@dataclass
class AgentState:
    """
    Runtime memory of the agent.
    Stores goal, constraints, and execution history.
    """

    goal: Optional[AgentGoal] = None
    current_step: Optional[str] = None
    history: List[str] = field(default_factory=list)

    def set_goal(self, goal: AgentGoal) -> None:
        """Attach a fully defined goal to state memory."""
        if not goal.is_fully_defined():
            raise ValueError("Cannot set undefined goal in state.")
        self.goal = goal
        self.history.append("Goal initialized with user constraints")

    def get_constraints(self) -> Optional[GoalConstraints]:
        """Access current constraints safely."""
        if self.goal:
            return self.goal.constraints
        return None

    def update_step(self, step_name: str) -> None:
        """Track planner progression."""
        self.current_step = step_name
        self.history.append(f"Transitioned to step: {step_name}")


# ----------------------
# Planner Component
# ----------------------

class PlanStep(Enum):
    """
    High-level actions the agent can take.
    These are NOT implementations, only decisions.
    """
    ASK_FOR_CONSTRAINTS = auto()
    DISCOVER_PRODUCTS = auto()
    CHECK_SATISFACTION = auto()


class AgentPlanner:
    """
    Decides what the agent should do next.
    """

    def next_step(self, goal: AgentGoal, state: AgentState) -> PlanStep:

        if not goal.is_fully_defined():
            return PlanStep.ASK_FOR_CONSTRAINTS

        return PlanStep.DISCOVER_PRODUCTS


# ----------------------
# Action Component
# ----------------------

class ProductDiscoveryEngine:
    """
    Uses OpenAI web-search tool to discover products
    dynamically from the internet.
    """

    def __init__(self, llm: LLMInterface):
        self.client = llm.client

    def discover(self, goal: AgentGoal) -> str:

        if not goal.is_fully_defined():
            raise ValueError("Goal must be fully defined before discovery.")

        constraints = goal.constraints

        search_query = f"""
        Find professional outfit items available online.

        Requirements:
        - Item name: {constraints.item_name}
        - Budget under ${constraints.budget}
        - Delivery within {constraints.delivery_deadline_days} days
        - Size: {constraints.size}
        - Style: {constraints.style}

        Return top 3 products according to constraints.
        """

        response = self.client.responses.create(
        model="gpt-4.1-mini",
        tools=[{
            "type": "web_search",
            "user_location": {
                "type": "approximate",
                "country": "PK"
            }
        }],
        input=search_query,
        )

        try:
            return response.output_text
        except Exception:
            return """⚠️ Required product is not mateched."""


# ---------------------
# Observations Component
# ---------------------

def check_user_satisfaction(user_input: str) -> bool:
    """
    Simple satisfaction detector.
    Returns True if user confirms satisfaction.
    """

    positive_signals = ["yes", "satisfied", "looks good", "perfect", "great", "ok", "okay"]
    user_input_lower = user_input.lower()

    return any(signal in user_input_lower for signal in positive_signals)
