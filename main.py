# ----------------------
# Imports
# ----------------------
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from agent_system import OPENAI_API_KEY, LLMInterface, AgentGoal, AgentState, AgentPlanner, PlanStep, ProductDiscoveryEngine, check_user_satisfaction

# Import your existing agent code here
# (Assume all your previous classes remain unchanged above this line)

# ----------------------
# FastAPI App
# ----------------------

app = FastAPI(title="Agentic Shopping Backend")

# In-memory session store (replace with Redis in production)
sessions: Dict[str, dict] = {}


# ----------------------
# Request / Response Models
# ----------------------

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    satisfied: bool
    constraints: dict


# ----------------------
# Session Creation
# ----------------------

@app.post("/session")
def create_session():
    session_id = str(uuid.uuid4())

    llm = LLMInterface(OPENAI_API_KEY)

    agent_goal = AgentGoal()
    agent_state = AgentState()
    planner = AgentPlanner()
    discovery_engine = ProductDiscoveryEngine(llm)

    sessions[session_id] = {
        "goal": agent_goal,
        "state": agent_state,
        "planner": planner,
        "discovery": discovery_engine,
        "llm": llm,
        "conversation": []
    }

    return {"session_id": session_id}


# ----------------------
# Chat Endpoint
# ----------------------

@app.post("/chat/{session_id}", response_model=ChatResponse)
def chat(session_id: str, request: ChatRequest):

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    agent_goal = session["goal"]
    agent_state = session["state"]
    planner = session["planner"]
    discovery_engine = session["discovery"]
    conversation_history = session["conversation"]
    llm = session["llm"]

    # Add user message
    conversation_history.append({"role": "user", "content": request.message})

    # Extract constraints
    llm_output = llm.conversational_constraint_parser(conversation_history)
    agent_goal.update_constraints_from_llm(llm_output)

    # Planning
    next_step = planner.next_step(agent_goal, agent_state)

    # --------------------------
    # Ask for missing constraints
    # --------------------------
    if next_step == PlanStep.ASK_FOR_CONSTRAINTS:

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
        conversation_history.append({"role": "assistant", "content": question})

        return ChatResponse(
            response=question,
            satisfied=False,
            constraints=agent_goal.constraints.__dict__ if agent_goal.constraints else {}
        )

    # --------------------------
    # Discover products
    # --------------------------
    agent_state.set_goal(agent_goal)
    agent_state.update_step("WEB_PRODUCT_DISCOVERY")

    results = discovery_engine.discover(agent_goal)

    if not results:
        return ChatResponse(
            response="No products found matching your constraints.",
            satisfied=False,
            constraints=agent_goal.constraints.__dict__
        )

    # --------------------------
    # Satisfaction check
    # --------------------------
    if check_user_satisfaction(request.message):
        return ChatResponse(
            response="Great! Happy shopping.",
            satisfied=True,
            constraints=agent_goal.constraints.__dict__
        )

    return ChatResponse(
        response=results + "\n\nAre you satisfied with these recommendations?",
        satisfied=False,
        constraints=agent_goal.constraints.__dict__
    )


# ----------------------
# Optional Debug Endpoint
# ----------------------

@app.get("/session/{session_id}")
def get_session_state(session_id: str):

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    return {
        "constraints": session["goal"].constraints.__dict__
        if session["goal"].constraints else {},
        "history": session["state"].history
    }
