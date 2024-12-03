import base64
import time
import warnings
from contextvars import ContextVar
from datetime import datetime
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
from tenacity import retry, stop_after_delay, wait_chain, wait_exponential, wait_fixed

from .modifiers.base import BaseModifier
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

    def get_or_create_project(self, name: str):
        response = requests.get(
            f"{self.server_url}/get_project", params={"project_name": name}
        )
        if response.status_code == 404:
            response = requests.post(
                f"{self.server_url}/create_project/", json={"name": name}
            )
            if response.status_code != 200:
                print(f"Error creating project: {response.text}")
                raise Exception("Failed to create project")
        elif response.status_code != 200:
            print(f"Error getting project: {response.text}")
            raise Exception("Failed to get project")
        return response.json()

    def project(self, name: str):
        return self.get_or_create_project(name)

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

    def use(
        self, function_id, modifier: Optional[BaseModifier] = None, run_async=False
    ):
        def wrapper(**inputs):
            if not run_async:
                # Original synchronous execution
                payload = {
                    "reasoner_id": function_id,
                    "inputs": base64.b64encode(cloudpickle.dumps(inputs)).decode(
                        "utf-8"
                    ),
                    "session_id": getattr(self, "_current_session", None),
                    "workflow_id": getattr(self, "_current_workflow", None),
                    "modifier": (
                        base64.b64encode(cloudpickle.dumps(modifier)).decode("utf-8")
                        if modifier
                        else None
                    ),
                }

                response = requests.post(
                    f"{self.server_url}/execute_reasoner/", json=payload
                )
                if response.status_code != 200:
                    raise Exception("Failed to execute function")

                response_data = response.json()
                if response_data.get("schema"):
                    try:
                        return create_dynamic_pydantic_model(response_data["schema"])(
                            **response_data["result"]
                        )
                    except:
                        warnings.warn(
                            f"Cannot create Pydantic model of type {response_data['schema']} from given response",
                            RuntimeWarning,
                        )
                # TODO: Make this return some standard pydantic model with the response result as a field so that we can standardize the response
                # NOTE: We will come here when modifier returns a schema that is not of type response_data["schema"] instead its own schema like for
                # translation, or gaurdrail etc. In that case we will not be able to create a pydantic model of the response_data["schema"]
                return response_data["result"]
            else:
                # Async execution
                future_id = str(uuid4())
                payload = {
                    "reasoner_id": function_id,
                    "inputs": base64.b64encode(cloudpickle.dumps(inputs)).decode(
                        "utf-8"
                    ),
                    "session_id": getattr(self, "_current_session", None),
                    "workflow_id": getattr(self, "_current_workflow", None),
                    "future_id": future_id,
                    "modifier": (
                        base64.b64encode(cloudpickle.dumps(modifier)).decode("utf-8")
                        if modifier
                        else None
                    ),
                }

                response = requests.post(
                    f"{self.server_url}/execute_reasoner_async/", json=payload
                )
                if response.status_code != 200:
                    raise Exception("Failed to execute function asynchronously")

                return Future(self, future_id)

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
        project_name = project["name"] if project else "Default Project"
        params = {"workflow_name": multiagent_name, "project_id": project_id}
        response = requests.get(f"{self.server_url}/list_runs", params=params)

        if response.status_code != 200:
            raise Exception("Failed to list runs")

        # Create the table with enhanced styling
        table_title = f"MultiAgent Session Runs in Project: {project_name}"
        table = Table(title=table_title, box=box.SIMPLE, show_lines=True)
        # Add columns with appropriate alignment and width
        table.add_column("Session ID", justify="left", overflow="fold", min_width=36)
        table.add_column("MultiAgent", justify="center")
        table.add_column("Reasoner Calls", justify="left", max_width=20)
        table.add_column("Start Date & Time", justify="center")
        table.add_column("Total Duration (s)", justify="center", max_width=19)

        # Sort sessions by start time (latest first)
        sessions = sorted(
            response.json()["sessions"], key=lambda x: x["start_time"], reverse=True
        )

        # Populate rows with formatted data
        for session in sessions:
            reasoner_calls = [
                f"{call['reasoner_name']} ({round(call['duration'], 2)}s)"
                for call in sorted(
                    session["reasoner_calls"], key=lambda x: x["timestamp"]
                )
            ]

            # Combine datetime into a human-readable format
            start_datetime = datetime.fromisoformat(session["start_time"]).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            table.add_row(
                session["session_id"],
                session["multiagent_name"],
                " → ".join(reasoner_calls),
                start_datetime,
                f"{round(session['total_duration'], 2)}",
            )

        # Render the table with a console
        console = Console(color_system="auto", width=200)
        console.print(table)

    def list_multiagents(self, project=None):
        project_id = project["project_id"] if project else None
        project_name = project["name"] if project else "Default Project"
        params = {"project_id": project_id}
        response = requests.get(f"{self.server_url}/list_multiagents", params=params)

        if response.status_code != 200:
            raise Exception("Failed to list multiagents")

        # Create the table with enhanced styling
        table_title = f"MultiAgents in Project: {project_name}"
        table = Table(title=table_title, box=box.SIMPLE, show_lines=True)
        table.add_column("MultiAgent", justify="left")
        table.add_column("ID", justify="left", overflow="fold", min_width=36)
        table.add_column("Tags", justify="left", max_width=20)
        table.add_column("Created At", justify="center")

        multiagents = response.json()["multiagents"]
        multiagents = sorted(
            response.json()["multiagents"], key=lambda x: x["created_at"], reverse=True
        )
        for multiagent in multiagents:
            created_at = datetime.fromisoformat(multiagent["created_at"]).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            table.add_row(
                multiagent["workflow_name"],
                multiagent["workflow_id"],
                ", ".join(multiagent["tags"]),
                created_at,
            )

        console = Console(color_system="auto", width=200)
        console.print(table)

    def list_reasoners(self, project=None):
        project_id = project["project_id"] if project else None
        project_name = project["name"] if project else "Default Project"
        params = {"project_id": project_id}
        response = requests.get(f"{self.server_url}/list_reasoners", params=params)

        if response.status_code != 200:
            raise Exception("Failed to list reasoners")

        # Create the table with enhanced styling
        table_title = f"Reasoners in Project: {project_name}"
        table = Table(title=table_title, box=box.SIMPLE, show_lines=True)
        table.add_column("Reasoner", justify="left")
        table.add_column("ID", justify="left", overflow="fold", min_width=36)
        table.add_column("Tags", justify="left", max_width=20)
        table.add_column("Created At", justify="center")

        reasoners = response.json()["reasoners"]
        reasoners = sorted(
            response.json()["reasoners"], key=lambda x: x["created_at"], reverse=True
        )
        for reasoner in reasoners:
            created_at = datetime.fromisoformat(reasoner["created_at"]).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            table.add_row(
                reasoner["reasoner_name"],
                reasoner["reasoner_id"],
                ", ".join(reasoner["tags"]),
                created_at,
            )

        console = Console(color_system="auto", width=200)
        console.print(table)


# ------ Class to represent a multi-agent workflow ------


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


# ---- Async Future class to represent a future result ----
class Future:
    def __init__(self, client, future_id):
        self.client = client
        self.future_id = future_id

    def get(self):
        return self.result()

    def result(self):
        """Blocking call to get result with exponential backoff"""

        @retry(
            stop=stop_after_delay(60),
            wait=wait_chain(
                *[wait_fixed(0.1) for i in range(50)]
                + [wait_fixed(0.5) for i in range(10)]
                + [wait_exponential(multiplier=1, min=0.1, max=10)]
            ),
        )
        def fetch_result():
            response = requests.get(
                f"{self.client.server_url}/get_future_result/{self.future_id}"
            )
            if response.status_code != 200:
                raise Exception(f"Failed to get future result: {response.text}")

            data = response.json()
            if data["status"] == "completed":
                result = data["result"]
                if result.get("schema"):
                    return create_dynamic_pydantic_model(result["schema"])(
                        **result["result"]
                    )
                return result["result"]
            raise Exception("Result not ready")

        return fetch_result()
