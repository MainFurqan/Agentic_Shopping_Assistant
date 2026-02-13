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
# LLM Definition or Interface
# -------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API")

client = OpenAI(api_key=OPENAI_API_KEY)


SYSTEM_PROMPT = """
You extract shopping constraints from conversation.

Return JSON ONLY.

If a value is mentioned, extract it.
If not mentioned, return null.

Format:

{
  "budget": number or null,
  "delivery_deadline_days": number or null,
  "size": string or null,
  "style": string or null
}
"""


def conversational_constraint_parser(conversation_history: List[Dict]) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history,
        temperature=0
    )

    return json.loads(response.choices[0].message.content)



# --------------------
# Goal Component
# --------------------

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
        if self.constraints is None:
            return False

        return (
            self.constraints.budget > 0 and
            self.constraints.delivery_deadline_days > 0 and
            self.constraints.size is not None and
            self.constraints.style is not None
        )

    

    def apply_llm_constraints(self, llm_output: Dict) -> None:
        """
        Reads constraint values from LLM JSON output and applies them
        to the goal using set_constraints().

        Raises:
            ValueError: If required fields are missing or invalid.
        """

        # Validate required fields
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
            budget=budget,
            delivery_deadline_days=delivery_deadline_days,
            size=size,
            style=style,
            must_have_categories=self.required_categories,
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


class AgentPlanner:
    """
    Decides what the agent should do next.
    """

class AgentPlanner:

    def next_step(self, goal: AgentGoal, state: AgentState) -> PlanStep:

        if not goal.is_fully_defined():
            return PlanStep.ASK_FOR_CONSTRAINTS

        return PlanStep.DISCOVER_PRODUCTS


# ----------------------
# Action Component
# ----------------------

class ProductDiscoveryEngine:
    """
    Handles product discovery and filtering
    across multiple mocked marketplaces.
    """

    def __init__(self, data_files: List[str]):
        self.data_files = data_files
        self.products = self._load_all_products()

    def _load_all_products(self) -> List[Dict]:
        all_products = []
        for file in self.data_files:
            with open(file, "r") as f:
                data = json.load(f)
                all_products.extend(data)
        return all_products

    def _extract_delivery_days(self, delivery_str: str) -> int:
        # "4 days" -> 4
        return int(delivery_str.split()[0])

    def discover(self, goal: AgentGoal) -> List[Dict]:
        """
        Returns filtered and ranked products
        according to goal constraints.
        """

        if not goal.is_fully_defined():
            raise ValueError("Goal must be fully defined before discovery.")

        constraints = goal.constraints
        required_categories = goal.required_categories

        filtered = []

        for product in self.products:

            # Category filter
            if product["category"] not in required_categories:
                continue

            # Budget filter
            if product["price"] > constraints.budget:
                continue

            # Delivery filter
            delivery_days = self._extract_delivery_days(product["delivery_estimate"])
            if delivery_days > constraints.delivery_deadline_days:
                continue

            # Basic style matching (naive string match)
            if constraints.style and constraints.style.lower() not in product["title"].lower():
                continue

            filtered.append(product)

        # Rank by price (ascending)
        filtered.sort(key=lambda x: x["price"])

        return filtered


# ----------------------
# Orchestration Component
# ----------------------
if __name__ == "__main__":

    # Initialize core components
    agent_goal = AgentGoal()
    agent_state = AgentState()
    planner = AgentPlanner()

    # Load mocked marketplace data
    discovery_engine = ProductDiscoveryEngine(
        data_files=[
            "data\\mock_amazon.json",
            "data\\mock_ebay.json",
            "data\\mock_walmart.json"
        ]
    )

    conversation_history = []

    print("\n👔 Professional Outfit Shopping Assistant")
    print("Tell me what you're looking for.\n")

    while True:

        # --------------------------
        # 1. Get user input
        # --------------------------
        user_message = input("User: ")
        conversation_history.append({"role": "user", "content": user_message})

        # --------------------------
        # 2. Extract constraints via LLM
        # --------------------------
        llm_output = conversational_constraint_parser(conversation_history)

        # Update constraints incrementally
        agent_goal.update_constraints_from_llm(llm_output)

        # --------------------------
        # 3. Planner Decision
        # --------------------------
        next_step = planner.next_step(agent_goal, agent_state)

        if next_step == PlanStep.ASK_FOR_CONSTRAINTS:

            # Determine missing fields
            missing = []
            c = agent_goal.constraints

            if c is None or c.budget <= 0:
                missing.append("budget")

            if c is None or c.delivery_deadline_days <= 0:
                missing.append("delivery deadline (in days)")

            if c is None or c.size is None:
                missing.append("size")

            if c is None or c.style is None:
                missing.append("style")

            question = "Please provide your " + ", ".join(missing) + "."
            print("\nAssistant:", question)

            conversation_history.append({
                "role": "assistant",
                "content": question
            })

            continue

        # --------------------------
        # 4. Goal is fully defined
        # --------------------------
        agent_state.set_goal(agent_goal)

        print("\n✅ Goal fully defined with constraints:")
        print(agent_state.get_constraints())

        # --------------------------
        # 5. Discover products
        # --------------------------
        agent_state.update_step("DISCOVER_PRODUCTS")

        results = discovery_engine.discover(agent_goal)

        if not results:
            print("\n❌ No products found matching your constraints.")
            break

        # --------------------------
        # 6. Show top results
        # --------------------------
        print("\n🎯 Top Recommended Products:\n")

        top_results = results[:5]  # Top 5

        for i, product in enumerate(top_results, start=1):
            print(f"{i}. {product['title']}")
            print(f"   Category: {product['category']}")
            print(f"   Price: ${product['price']}")
            print(f"   Delivery: {product['delivery_estimate']}")
            print()

        print("🛍️ Recommendation complete.")
        break
