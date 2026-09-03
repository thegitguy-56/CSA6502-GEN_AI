"""
IndustroSense AI - Knowledge Retrieval Agent Module
Implements a simple rule-based ReAct-style agent with two tools:
  1. retrieve_tool  - semantic search over the RAG vector store
  2. schedule_tool  - deterministic maintenance-schedule calculator

The agent exposes a full decision trace (Thought / Action / Observation)
for auditability, per the assignment's Responsible-Deployment requirement.
"""
import re
import json

SCHEDULE_INTERVALS_HOURS = {
    "bearing_grease": 4000,
    "bearing_replacement": 18000,
    "seal_inspection": 8000,
    "seal_replacement": 21000,
    "alignment_check": 4000,
}

SCHEDULE_KEYWORDS = ["due", "overdue", "next service", "schedule", "how many hours",
                     "interval", "when should", "service hour"]


def schedule_tool(component, last_service_hours, current_operating_hours):
    """Deterministic maintenance-schedule calculator tool."""
    interval = SCHEDULE_INTERVALS_HOURS.get(component)
    if interval is None:
        return {"error": f"Unknown component '{component}'. Known: {list(SCHEDULE_INTERVALS_HOURS)}"}
    next_service = last_service_hours + interval
    overdue_by = current_operating_hours - next_service
    status = "OVERDUE" if overdue_by > 0 else "DUE SOON" if (next_service - current_operating_hours) < 500 else "OK"
    return {
        "component": component,
        "interval_hours": interval,
        "next_service_hours": next_service,
        "current_operating_hours": current_operating_hours,
        "overdue_by_hours": max(overdue_by, 0),
        "status": status,
    }


def retrieve_tool(store, query, k=3):
    return store.search(query, k=k)


class IndustroSenseAgent:
    """
    Decision policy: rule-based router (lightweight ReAct-style loop).
    Thought -> decide which tool(s) to call -> Action -> Observation -> Final Answer.
    Falls back to a clarifying question if the query is ambiguous (no schedule
    numbers given but a schedule question is asked).
    """

    def __init__(self, store):
        self.store = store

    def route(self, query):
        trace = []
        q_lower = query.lower()
        wants_schedule = any(kw in q_lower for kw in SCHEDULE_KEYWORDS)

        trace.append({"step": "Thought", "content":
                       f"Query mentions schedule/interval keywords: {wants_schedule}. "
                       f"Decide whether to call schedule_tool, retrieve_tool, or both."})

        result = {"query": query, "trace": trace, "retrieved": [], "schedule_result": None,
                  "clarifying_question": None}

        if wants_schedule:
            hours = [int(x) for x in re.findall(r"\b(\d{3,6})\b", query)]
            component = None
            for comp in SCHEDULE_INTERVALS_HOURS:
                key = comp.replace("_", " ").split()[0]
                if key in q_lower or "bearing" in q_lower and "bearing" in comp:
                    component = comp
            if "seal" in q_lower:
                component = "seal_replacement" if "replace" in q_lower else "seal_inspection"
            elif "bearing" in q_lower:
                component = "bearing_replacement" if "replac" in q_lower else "bearing_grease"
            elif "alignment" in q_lower:
                component = "alignment_check"

            if component and len(hours) >= 2:
                trace.append({"step": "Action", "content":
                               f"Call schedule_tool(component='{component}', last_service_hours={hours[0]}, "
                               f"current_operating_hours={hours[1]})"})
                sched = schedule_tool(component, hours[0], hours[1])
                trace.append({"step": "Observation", "content": str(sched)})
                result["schedule_result"] = sched
                # Also ground the numeric answer with retrieved policy text for context
                trace.append({"step": "Thought", "content":
                               "Also retrieve supporting reference text for interval justification."})
                retrieved = retrieve_tool(self.store, query, k=2)
                trace.append({"step": "Action", "content": f"Call retrieve_tool(query='{query}', k=2)"})
                trace.append({"step": "Observation", "content":
                               f"Retrieved {len(retrieved)} supporting chunks: " +
                               ", ".join(r['chunk_id'] for r in retrieved)})
                result["retrieved"] = retrieved
            else:
                trace.append({"step": "Action", "content":
                               "Insufficient numeric fields (need component, last_service_hours, "
                               "current_operating_hours) to call schedule_tool."})
                trace.append({"step": "Observation", "content":
                               "Missing structured inputs; cannot compute schedule deterministically."})
                result["clarifying_question"] = (
                    "To compute the maintenance schedule, please provide: component name "
                    "(bearing/seal/alignment), last service hour reading, and current operating hours."
                )
                trace.append({"step": "Final Decision", "content": "Ask clarifying question (no guessing on safety-relevant numbers)."})
        else:
            trace.append({"step": "Action", "content": f"Call retrieve_tool(query='{query}', k=3)"})
            retrieved = retrieve_tool(self.store, query, k=3)
            trace.append({"step": "Observation", "content":
                           f"Retrieved {len(retrieved)} chunks: " + ", ".join(r['chunk_id'] for r in retrieved)})
            result["retrieved"] = retrieved
            trace.append({"step": "Final Decision", "content": "Sufficient grounding found; compose RAG-augmented answer."})

        return result
