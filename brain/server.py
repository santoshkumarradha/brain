import base64
import json
import uuid
from datetime import datetime, timezone

import cloudpickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
from tinydb import Query, TinyDB

from .llm import OpenAILLM
from .schema import convert_prompt
from .utils import schema_to_pydantic_class

app = FastAPI()
llm = OpenAILLM(model_name="gpt-4o-mini")
# TinyDB databases
reasoner_db = TinyDB("reasoner_registry.json")
lineage_db = TinyDB("lineage_registry.json")


class MultiModal(BaseModel):
    text: str = None
    schema: dict | None = None


class RegisterRequest(BaseModel):
    reasoner_name: str
    reasoner_code: str  # base64 encoded pickled reasoner
    schema: dict | None = None  # JSON representation of the Pydantic model schema


class ExecuteRequest(BaseModel):
    reasoner_id: str
    inputs: str  # base64 encoded pickled inputs
    session_id: str | None = None  # Optional session context


@app.post("/register_reasoner/")
async def register_reasoner(request: RegisterRequest):
    reasoner_name = request.reasoner_name

    # Create a unique reasoner ID
    reasoner_id = f"{reasoner_name}_v{len(reasoner_db) + 1}"
    Reasoner = Query()
    if reasoner_db.search(Reasoner.name == reasoner_name):
        reasoner_id = reasoner_name  # Re-use the existing reasoner name

    reasoner_db.insert(
        {
            "reasoner_id": reasoner_id,
            "reasoner_name": reasoner_name,
            "reasoner_code": request.reasoner_code,
            "schema": request.schema,
        }
    )
    return {"reasoner_id": reasoner_id}


@app.post("/execute_reasoner/")
async def execute_reasoner(request: ExecuteRequest):
    reasoner_id = request.reasoner_id
    Reasoner = Query()
    result = reasoner_db.search(Reasoner.reasoner_id == reasoner_id)
    if not result:
        raise HTTPException(status_code=404, detail="Reasoner not found")

    reasoner = cloudpickle.loads(base64.b64decode(result[0]["reasoner_code"]))
    inputs = cloudpickle.loads(base64.b64decode(request.inputs))

    # Capture start time
    start_time = datetime.now(timezone.utc)

    # Execute the reasoner
    llm_input = reasoner(**inputs)
    llm_input = convert_prompt(llm_input)

    schema_dict = result[0].get("schema")
    schema = schema_to_pydantic_class(schema_dict)

    # Generate response
    response = llm.generate(prompt=llm_input.format(), schema=schema)

    # Capture stop time
    stop_time = datetime.now(timezone.utc)

    # Calculate duration
    duration = (stop_time - start_time).total_seconds()

    # Store lineage information if session_id is present
    if request.session_id:
        lineage_db.insert(
            {
                "session_id": request.session_id,
                "reasoner_id": reasoner_id,
                "inputs": str(inputs),  # Store string representation of inputs
                "result": str(response),  # Store string representation of result
                "timestamp": start_time.isoformat(),
                "stop_time": stop_time.isoformat(),  # Stop time
                "duration": duration,  # Duration in seconds
            }
        )

    return {"result": response, "schema": schema_dict}


@app.post("/create_session/")
async def create_session():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


@app.get("/get_call_graph/{session_id}")
async def get_call_graph(session_id: str):
    Call = Query()
    calls = lineage_db.search(Call.session_id == session_id)
    if not calls:
        raise HTTPException(status_code=404, detail="Session ID not found")
    return {"session_id": session_id, "lineage": calls}
