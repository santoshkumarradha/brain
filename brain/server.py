import asyncio
import base64
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import cloudpickle
from fastapi import BackgroundTasks, FastAPI, HTTPException
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
# TODO in-memory future storage we can move this to a database later
future_registry: Dict[str, asyncio.Future] = {}


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
    modifier: str | None = None  # Add modifier field base64 encoded


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
    response, schema_dict, start_time, duration = await _execute_reasoner_core(
        reasoner_id=request.reasoner_id,
        inputs=request.inputs,
        modifier=request.modifier,
        is_async=False,
    )

    Reasoner = Query()
    reasoner_info = reasoner_db.search(Reasoner.reasoner_id == request.reasoner_id)[0]

    _store_lineage(
        session_id=request.session_id,
        reasoner_id=request.reasoner_id,
        workflow_id=request.workflow_id,
        inputs=request.inputs,
        response=response,
        start_time=start_time,
        duration=duration,
        reasoner_info=reasoner_info,
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
    existing_project = project_db.get(Project.name == request.name)
    if existing_project:
        return existing_project

    project_id = str(uuid4())
    project = {
        "project_id": project_id,
        "name": request.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    project_db.insert(project)
    return project


async def get_project(
    project_name: Optional[str] = None, project_id: Optional[str] = None
):
    if project_id:
        project = project_db.get(Query().project_id == project_id)
    elif project_name:
        project = project_db.get(Query().project_name == project_name)
    else:
        raise HTTPException(
            status_code=400, detail="Project name or ID must be provided"
        )

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return project


@app.get("/get_project")
async def get_project_endpoint(
    project_name: Optional[str] = None, project_id: Optional[str] = None
):
    project = await get_project(project_name, project_id)
    return project


@app.get("/list_runs")
async def list_runs(
    workflow_name: Optional[str] = None, project_id: Optional[str] = None
):
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


@app.get("/list_reasoners")
async def list_reasoners(project_id: Optional[str] = None):
    Query_filter = Query()
    if project_id:
        conditions = Query_filter.project_id == project_id
    else:
        conditions = Query_filter.project_id.exists()
    reasoners = reasoner_db.search(conditions)
    return {"reasoners": reasoners}


@app.get("/list_multiagents")
async def list_multiagents(project_id: Optional[str] = None):
    Query_filter = Query()
    if project_id:
        conditions = Query_filter.project_id == project_id
    else:
        conditions = Query_filter.project_id.exists()
    multiagents = workflow_db.search(conditions)
    return {"multiagents": multiagents}


# ------ Async Execution ------


def _store_lineage(
    session_id: str,
    reasoner_id: str,
    workflow_id: Optional[str],
    inputs: str,
    response: Any,
    start_time: datetime,
    duration: float,
    reasoner_info: dict,
):
    """Store execution lineage information"""
    if not session_id:
        return

    lineage_db.insert(
        {
            "session_id": session_id,
            "reasoner_id": reasoner_id,
            "reasoner_name": reasoner_info["reasoner_name"],
            "workflow_id": workflow_id,
            "project_id": reasoner_info["project_id"],
            "inputs": str(inputs),
            "result": str(response),
            "timestamp": start_time.isoformat(),
            "stop_time": (start_time + timedelta(seconds=duration)).isoformat(),
            "duration": duration,
        }
    )


async def _execute_reasoner_core(
    reasoner_id: str,
    inputs: str,
    modifier: Optional[str] = None,
    is_async: bool = False,
) -> Tuple[Any, dict, datetime, float]:
    """
    Core execution logic shared between sync and async execution paths.
    Returns (response, schema_dict, start_time, duration)
    """
    Reasoner = Query()
    result = reasoner_db.search(Reasoner.reasoner_id == reasoner_id)
    if not result:
        raise HTTPException(status_code=404, detail="Reasoner not found")

    reasoner = cloudpickle.loads(base64.b64decode(result[0]["reasoner_code"]))
    decoded_inputs = cloudpickle.loads(base64.b64decode(inputs))
    decoded_modifier = (
        cloudpickle.loads(base64.b64decode(modifier)) if modifier else None
    )

    # Capture start time
    start_time = datetime.now(timezone.utc)

    # Execute the reasoner
    llm_input = reasoner(**decoded_inputs)
    llm_input = convert_prompt(llm_input)

    schema_dict = result[0].get("schema")
    schema = create_dynamic_pydantic_model(schema_dict)

    # Generate response using modifier if provided
    # TODO: Somehow we should be able to combine the two if conditions and have a default modifier that just calls llm.generate
    if decoded_modifier:
        if is_async:
            response = await decoded_modifier.async_modify(
                input=llm_input, schema=schema, model=llm
            )
        else:
            response = decoded_modifier.modify(
                input=llm_input, schema=schema, model=llm
            )
    else:
        response = (
            await llm.generate_async(prompt=llm_input.format(), schema=schema)
            if is_async
            else llm.generate(prompt=llm_input.format(), schema=schema)
        )

    # Calculate duration
    duration = (datetime.now(timezone.utc) - start_time).total_seconds()

    return response, schema_dict, start_time, duration


class AsyncExecuteRequest(BaseModel):
    reasoner_id: str
    workflow_id: str | None = None
    inputs: str  # base64 encoded pickled inputs
    session_id: str | None = None
    future_id: str  # To track this specific async request
    modifier: str | None = None  # base64 encoded pickled modifier


async def execute_reasoner_background(
    request: AsyncExecuteRequest, future: asyncio.Future
):
    try:
        response, schema_dict, start_time, duration = await _execute_reasoner_core(
            reasoner_id=request.reasoner_id,
            inputs=request.inputs,
            modifier=request.modifier,
            is_async=True,
        )

        Reasoner = Query()
        reasoner_info = reasoner_db.search(Reasoner.reasoner_id == request.reasoner_id)[
            0
        ]

        _store_lineage(
            session_id=request.session_id,
            reasoner_id=request.reasoner_id,
            workflow_id=request.workflow_id,
            inputs=request.inputs,
            response=response,
            start_time=start_time,
            duration=duration,
            reasoner_info=reasoner_info,
        )

        future.set_result({"result": response, "schema": schema_dict})

    except Exception as e:
        future.set_exception(e)


@app.post("/execute_reasoner_async/")
async def execute_reasoner_async(
    request: AsyncExecuteRequest, background_tasks: BackgroundTasks
):
    future_id = request.future_id
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    future_registry[future_id] = future

    background_tasks.add_task(execute_reasoner_background, request, future)

    return {"future_id": future_id}


@app.get("/get_future_result/{future_id}")
async def get_future_result(future_id: str):
    if future_id not in future_registry:
        raise HTTPException(status_code=404, detail="Future not found")

    future = future_registry[future_id]
    if not future.done():
        return {"status": "pending"}

    try:
        result = future.result()
        del future_registry[future_id]  # Cleanup
        return {"status": "completed", "result": result}
    except Exception as e:
        del future_registry[future_id]  # Cleanup
        raise HTTPException(status_code=500, detail=str(e))
