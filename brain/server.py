import base64
import json
import uuid

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

    # run the reasoner to get the multimodal input
    llm_input = reasoner(**inputs)

    # Validate and convert llm_input
    llm_input = convert_prompt(llm_input)

    # Load schema from the database for the given reasoner_id
    schema_dict = reasoner_db.search(Reasoner.reasoner_id == reasoner_id)[0].get(
        "schema"
    )
    schema = schema_to_pydantic_class(schema_dict)

    # Generate multimodal response
    result = llm.generate(prompt=llm_input.format(), schema=schema)

    return dict(result=result, schema=schema_dict)
