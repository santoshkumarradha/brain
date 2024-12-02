import base64
import json
import uuid
from datetime import datetime, timezone
from typing import List
from uuid import uuid4

import cloudpickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
from tinydb import Query, TinyDB

from .llm import OpenAILLM
from .schema import convert_prompt
from .utils import create_dynamic_pydantic_model

app = FastAPI()
llm = OpenAILLM(model_name="gpt-4o-mini")
# TinyDB databases
project_db = TinyDB("project_registry.json")
reasoner_db = TinyDB("reasoner_registry.json")
workflow_db = TinyDB("workflow_registry.json")
lineage_db = TinyDB("lineage_registry.json")


class MultiModal(BaseModel):
    text: str = None
    schema: dict | None = None


class ProjectCreate(BaseModel):
    name: str


class RegisterRequest(BaseModel):
    reasoner_name: str
    reasoner_code: str
    schema: dict = None
    project_id: str
    tags: List[str] = []


class WorkflowRegisterRequest(BaseModel):
    workflow_name: str
    workflow_code: str
    project_id: str
    tags: List[str] = []


class ExecuteRequest(BaseModel):
    reasoner_id: str
    workflow_id: str | None = None  # Add workflow ID
    inputs: str  # base64 encoded pickled inputs
    session_id: str | None = None  # Optional session context


@app.post("/register_reasoner/")
async def register_reasoner(request: RegisterRequest):
    Project = Query()
    if not project_db.search(Project.project_id == request.project_id):
        raise HTTPException(status_code=404, detail="Project not found")

    reasoner_id = str(uuid4())
    reasoner_db.insert(
        {
            "reasoner_id": reasoner_id,
            "reasoner_name": request.reasoner_name,
            "reasoner_code": request.reasoner_code,
            "schema": request.schema,
            "project_id": request.project_id,
            "tags": request.tags,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return {"reasoner_id": reasoner_id}


@app.post("/register_workflow/")
async def register_workflow(request: WorkflowRegisterRequest):
    Project = Query()
    if not project_db.search(Project.project_id == request.project_id):
        raise HTTPException(status_code=404, detail="Project not found")

    workflow_id = str(uuid4())
    workflow_db.insert(
        {
            "workflow_id": workflow_id,
            "workflow_name": request.workflow_name,
            "workflow_code": request.workflow_code,
            "project_id": request.project_id,
            "tags": request.tags,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return {"workflow_id": workflow_id}


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
    schema = create_dynamic_pydantic_model(schema_dict)

    # Generate response
    response = llm.generate(prompt=llm_input.format(), schema=schema)
    print(response)
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
                "reasoner_name": result[0]["reasoner_name"],
                "workflow_id": request.workflow_id,
                "project_id": result[0]["project_id"],
                "inputs": str(inputs),
                "result": str(response),
                "timestamp": start_time.isoformat(),
                "stop_time": stop_time.isoformat(),
                "duration": duration,
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


@app.post("/get_or_create_default_project/")
async def get_or_create_default_project():
    Project = Query()
    default_project = project_db.search(Project.name == "workspace")
    if default_project:
        return default_project[0]

    project_id = str(uuid4())
    project = {
        "project_id": project_id,
        "name": "workspace",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    project_db.insert(project)
    return project


@app.post("/create_project/")
async def create_project(request: ProjectCreate):
    Project = Query()
    if project_db.search(Project.name == request.name):
        raise HTTPException(status_code=400, detail="Project name already exists")

    project_id = str(uuid4())
    project = {
        "project_id": project_id,
        "name": request.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    project_db.insert(project)
    return project


@app.get("/list_runs")
async def list_runs(workflow_name: str | None = None, project_id: str | None = None):
    if not project_id:
        project = await get_or_create_default_project()
        project_id = project["project_id"]

    Query_filter = Query()
    conditions = Query_filter.project_id == project_id

    if workflow_name:
        workflow = workflow_db.get(
            (Query_filter.workflow_name == workflow_name)
            & (Query_filter.project_id == project_id)
        )
        if workflow:
            conditions &= Query_filter.workflow_id == workflow["workflow_id"]

    sessions = {}
    for run in lineage_db.search(conditions):
        session_id = run["session_id"]
        if session_id not in sessions:
            workflow = workflow_db.get(Query().workflow_id == run["workflow_id"])
            sessions[session_id] = {
                "session_id": session_id,
                "multiagent_name": (
                    workflow["workflow_name"] if workflow else "Direct Call"
                ),
                "reasoner_calls": [],
                "start_time": run["timestamp"],
                "total_duration": 0,
            }

        sessions[session_id]["reasoner_calls"].append(
            {
                "reasoner_name": run["reasoner_name"],
                "timestamp": run["timestamp"],
                "duration": run["duration"],
            }
        )
        sessions[session_id]["total_duration"] += run["duration"]

    return {"sessions": list(sessions.values())}
