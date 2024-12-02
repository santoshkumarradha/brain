from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from ...schema import MultiModalInput, TextPromptInput
from ..base import BaseModifier


class COT(BaseModifier):
    def __init__(self, num_steps=None):
        self.num_steps = num_steps

    def get_cot_schema(self, num_steps: Optional[int], schema: Type[BaseModel]):
        class Thought(BaseModel):
            thought: str = Field(
                ..., description="A single step in the chain of thought"
            )

        if num_steps is not None:
            # Prepare dynamic fields with proper type annotations
            annotations: Dict[str, Any] = {
                f"thought_{i}": Thought for i in range(1, num_steps + 1)
            }
            annotations["final_thought"] = schema

            # Prepare the attributes (fields) for the class
            fields = {
                f"thought_{i}": Field(
                    ..., description=f"Step {i} in the chain of thought"
                )
                for i in range(1, num_steps + 1)
            }
            fields["final_thought"] = Field(
                ..., description="Final conclusion or output"
            )

            # Create the class dynamically
            Steps = type(
                "Steps",
                (BaseModel,),
                {
                    "__annotations__": annotations,
                    **fields,
                },
            )
        else:
            # Create a class with a list of thoughts and a final thought

            class Steps(BaseModel):
                thoughts: List[Thought] = Field(
                    ..., description="A list of steps in the chain of thought"
                )
                final_thought: schema = Field(
                    ..., description="Final conclusion or output"
                )

        return Steps

    def modify(
        self, input: MultiModalInput, schema: BaseModel, model: Any
    ) -> BaseModel:
        input.text.system_prompt += (
            "\n\nPlease think step by step and provide a chain of thought reasoning "
            "with multiple steps with final step being the answer."
        )
        cot_schema = self.get_cot_schema(self.num_steps, schema)
        response = model.generate(input.format(), schema=cot_schema)
        print(response)
        return response.final_thought
