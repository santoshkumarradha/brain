import base64
from contextvars import ContextVar
from functools import wraps
from typing import Any, Dict, List, Optional
from uuid import uuid4

import cloudpickle
import pyperclip
import requests
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .utils import create_dynamic_pydantic_model

# Thread-local storage for session context
current_session_id: ContextVar[str] = ContextVar("current_session_id", default=None)


class BrainClient:
    def __init__(self, server_url):
        self.server_url = server_url
        self._default_project = self.get_or_create_default_project()

    def get_or_create_default_project(self):
        response = requests.post(f"{self.server_url}/get_or_create_default_project/")
        if response.status_code != 200:
            raise Exception("Failed to initialize default project")
        return response.json()

    def create_project(self, name: str):
        response = requests.post(
            f"{self.server_url}/create_project/", json={"name": name}
        )
        if response.status_code != 200:
            raise Exception("Failed to create project")
        return response.json()

    def register(self, func, schema=None, project=None, name=None, tags=None):
        function_code = base64.b64encode(cloudpickle.dumps(func)).decode("utf-8")
        project_id = (
            project["project_id"] if project else self._default_project["project_id"]
        )

        response = requests.post(
            f"{self.server_url}/register_reasoner/",
            json={
                "reasoner_name": name or func.__name__,
                "reasoner_code": function_code,
                "schema": schema.schema() if schema else None,
                "project_id": project_id,
                "tags": tags or [],
            },
        )
        if response.status_code != 200:
            raise Exception("Failed to register function")
        return response.json()["reasoner_id"]

    def register_workflow(self, func, project=None, tags=None, name=None):
        try:
            payload = {
                "workflow_name": name or func.__name__,
                "workflow_code": base64.b64encode(cloudpickle.dumps(func)).decode(
                    "utf-8"
                ),
                "project_id": (
                    project["project_id"]
                    if project
                    else self._default_project["project_id"]
                ),
                "tags": tags or [],
            }

            response = requests.post(
                f"{self.server_url}/register_workflow/", json=payload
            )
            if response.status_code != 200:
                print(f"Error response: {response.text}")  # Debug print
                raise Exception("Failed to register workflow")
            return response.json()["workflow_id"]
        except Exception as e:
            print(f"Exception during serialization: {str(e)}")  # Debug print
            raise

    def use(self, function_id):
        def wrapper(**inputs):
            payload = {
                "reasoner_id": function_id,
                "inputs": base64.b64encode(cloudpickle.dumps(inputs)).decode("utf-8"),
                "session_id": getattr(self, "_current_session", None),
                "workflow_id": getattr(self, "_current_workflow", None),
            }

            response = requests.post(
                f"{self.server_url}/execute_reasoner/", json=payload
            )
            if response.status_code != 200:
                raise Exception("Failed to execute function")

            response_data = response.json()
            if response_data.get("schema"):
                return create_dynamic_pydantic_model(response_data["schema"])(
                    **response_data["result"]
                )
            return response_data

        return wrapper

    def multi_agent(self, tags=None, project=None, name=None, auto_register=True):
        if callable(tags):  # @multi_agent case
            func = tags
            wrapper = wraps(func)(func)
            wrapper.id = None
            if auto_register:
                wrapper.id = self.register_workflow(func, None, [], func.__name__)
            wrapper.register = lambda: self.register_workflow(
                func, None, [], func.__name__
            )
            return MultiAgent(func, wrapper.id, self)

        def decorator(func):
            wrapper = wraps(func)(func)
            wrapper.id = None
            if auto_register:
                wrapper.id = self.register_workflow(
                    func, project, list(tags) if tags else [], name
                )
            wrapper.register = lambda: self.register_workflow(
                func, project, list(tags) if tags else [], name
            )
            return MultiAgent(func, wrapper.id, self)

        return decorator

        def get_call_graph(self, session_id):
            response = requests.get(f"{self.server_url}/get_call_graph/{session_id}")
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception("Failed to fetch call graph")

    def reasoner(
        self, name=None, schema=None, project=None, tags=None, auto_register=True
    ):
        def decorator(func):
            wrapper = wraps(func)(func)
            wrapper.id = None

            if auto_register:
                wrapper.id = self.register(
                    func, schema, project, name, list(tags) if tags else []
                )

            wrapper.register = lambda: self.register(
                func, schema, project, name, list(tags) if tags else []
            )
            return wrapper

        if callable(name):  # @reasoner case
            return decorator(name)
        return decorator

    def get_call_graph(self, session_id):
        response = requests.get(f"{self.server_url}/get_call_graph/{session_id}")
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception("Failed to fetch call graph")

    def _create_session(self):
        """Helper to create new session"""
        response = requests.post(f"{self.server_url}/create_session/")
        if response.status_code != 200:
            raise Exception("Failed to create session")
        return response.json()

    def list_runs(self, multiagent_name: str = None, project=None):
        project_id = project["project_id"] if project else None
        params = {"workflow_name": multiagent_name, "project_id": project_id}
        response = requests.get(f"{self.server_url}/list_runs", params=params)

        if response.status_code != 200:
            raise Exception("Failed to list runs")

        # Create the table with enhanced styling
        table = Table(title="MultiAgent Session Runs", box=box.SIMPLE, show_lines=True)

        # Add columns with appropriate alignment and width
        table.add_column("Session ID", justify="left", overflow="fold", min_width=36)
        table.add_column("MultiAgent", justify="center")
        table.add_column("Reasoner Calls", justify="left", max_width=20)
        table.add_column("Start Date", justify="center")
        table.add_column("Start Time", justify="center")
        table.add_column("Total Duration (s)", justify="center", max_width=19)

        # Populate rows with formatted data
        for session in response.json()["sessions"]:
            reasoner_calls = [
                f"{call['reasoner_name']} ({round(call['duration'], 2)}s)"
                for call in sorted(
                    session["reasoner_calls"], key=lambda x: x["timestamp"]
                )
            ]

            # Split datetime into date and time
            start_datetime = session["start_time"].split("T")
            start_date = start_datetime[0]
            start_time = start_datetime[1] if len(start_datetime) > 1 else ""

            table.add_row(
                session["session_id"],
                session["multiagent_name"],
                " → ".join(reasoner_calls),
                start_date,
                start_time,
                f"{round(session['total_duration'], 2)}",
            )

        # Render the table with a console
        console = Console(color_system="auto", width=200)
        console.print(table)


class MultiAgent:
    def __init__(self, func, workflow_id, brain_client):
        self.func = func
        self.workflow_id = workflow_id
        self.brain_client = brain_client
        self.__name__ = func.__name__

    def __call__(self, *args, **kwargs):
        session = self.brain_client._create_session()

        # Set context for this run
        self.brain_client._current_session = session["session_id"]
        self.brain_client._current_workflow = self.workflow_id

        try:
            return self.func(*args, **kwargs)
        finally:
            self.brain_client._current_session = None
            self.brain_client._current_workflow = None

    def _execute(self, session_id, *args, **kwargs):
        # Store session/workflow context for this run
        self.brain_client._current_session = session_id
        self.brain_client._current_workflow = self.workflow_id
        try:
            return self.func(*args, **kwargs)
        finally:
            # Clear context after run
            self.brain_client._current_session = None
            self.brain_client._current_workflow = None
