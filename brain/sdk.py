import base64
from contextvars import ContextVar
from functools import wraps
from uuid import uuid4

import cloudpickle
import requests
from pydantic import create_model

from .utils import create_dynamic_pydantic_model

# Thread-local storage for session context
current_session_id: ContextVar[str] = ContextVar("current_session_id", default=None)


class BrainClient:
    def __init__(self, server_url):
        self.server_url = server_url

    def register(self, func, schema=None):
        function_code = base64.b64encode(cloudpickle.dumps(func)).decode("utf-8")
        function_name = func.__name__
        schema_dict = schema.schema() if schema else None
        response = requests.post(
            f"{self.server_url}/register_reasoner/",
            json={
                "reasoner_name": function_name,
                "reasoner_code": function_code,
                "schema": schema_dict,
            },
        )
        if response.status_code == 200:
            return response.json()["reasoner_id"]
        else:
            raise Exception("Failed to register function")

    def use(self, function_id):
        def wrapper(**inputs):
            # Automatically fetch session ID from context
            session_id = current_session_id.get()
            inputs_encoded = base64.b64encode(cloudpickle.dumps(inputs)).decode("utf-8")
            payload = {"reasoner_id": function_id, "inputs": inputs_encoded}
            if session_id:
                payload["session_id"] = session_id
            response = requests.post(
                f"{self.server_url}/execute_reasoner/", json=payload
            )
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("schema"):
                    return create_dynamic_pydantic_model(response_data["schema"])(
                        **response_data["result"]
                    )
                return response_data
            else:
                raise Exception("Failed to execute function")

        return wrapper

    def multi_agent(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a new session for this multi-agent call
            session_response = requests.post(f"{self.server_url}/create_session/")
            if session_response.status_code != 200:
                raise Exception("Failed to create session")
            session_id = session_response.json()["session_id"]

            # Set session ID in thread-local storage
            token = current_session_id.set(session_id)
            try:
                # Call the original function
                return func(*args, **kwargs)
            finally:
                # Reset the session ID after the function completes
                current_session_id.reset(token)

        return wrapper

    def get_call_graph(self, session_id):
        response = requests.get(f"{self.server_url}/get_call_graph/{session_id}")
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception("Failed to fetch call graph")

    def reasoner(self, schema=None):
        def decorator(func):
            def wrapper():
                return func

            wrapper.register = lambda: self.register(func, schema)
            return wrapper

        return decorator
