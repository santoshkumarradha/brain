import base64

import cloudpickle
import requests
from pydantic import BaseModel, create_model

from .utils import schema_to_pydantic_class


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
            # Encode the inputs for the server
            inputs_encoded = base64.b64encode(cloudpickle.dumps(inputs)).decode("utf-8")
            response = requests.post(
                f"{self.server_url}/execute_reasoner/",
                json={"reasoner_id": function_id, "inputs": inputs_encoded},
            )
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("schema"):
                    return schema_to_pydantic_class(response_data["schema"])(
                        **response_data["result"]
                    )
                return response_data
            else:
                raise Exception("Failed to execute function")

        return wrapper

    def reasoner(self, schema=None):
        def decorator(func):
            def wrapper():
                return func

            wrapper.register = lambda: self.register(func, schema)
            return wrapper

        return decorator
