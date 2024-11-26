from pydantic import BaseModel, Field


class TextPromptInput(BaseModel):
    system_prompt: str = Field(..., description="The system prompt")
    user_prompt: str = Field(..., description="The user prompt")


class MultiModalInput(BaseModel):
    text: TextPromptInput = Field(..., description="The text prompt input")

    def format(self):
        return [
            {"role": "system", "content": self.text.system_prompt},
            {"role": "user", "content": self.text.user_prompt},
        ]


def convert_prompt(llm_input):
    """
    Converts input to MultiModalInput type. Validates the input type and raises an error if invalid.
    Supports:
    - MultiModalInput
    - Tuple (user_prompt[str], system_prompt[str])
    - Dict with keys "user_prompt" and optional "system_prompt"
    - user_prompt[str]
    """
    if isinstance(llm_input, MultiModalInput):
        return llm_input

    try:
        if isinstance(llm_input, tuple) and len(llm_input) == 2:
            user_prompt, system_prompt = llm_input
            return MultiModalInput(
                text=TextPromptInput(
                    system_prompt=system_prompt, user_prompt=user_prompt
                )
            )
        elif isinstance(llm_input, dict):
            user_prompt = llm_input.get("user_prompt")
            system_prompt = llm_input.get(
                "system_prompt", "You are an AI assistant helping a user with a task."
            )
            if not user_prompt or not isinstance(user_prompt, str):
                raise ValueError(
                    "Dictionary input must contain a 'user_prompt' key with a string value."
                )
            return MultiModalInput(
                text=TextPromptInput(
                    system_prompt=system_prompt, user_prompt=user_prompt
                )
            )
        elif isinstance(llm_input, str):
            return MultiModalInput(
                text=TextPromptInput(
                    system_prompt="You are an AI assistant helping a user with a task.",
                    user_prompt=llm_input,
                )
            )
    except Exception as e:
        print(type(llm_input))
        raise ValueError(
            f"Input must be of type MultiModalInput, (user_prompt[str], system_prompt[str]), "
            f"dict with 'user_prompt' and optional 'system_prompt', or user_prompt[str]. "
            f"Received type: {type(llm_input)}"
        ) from e

    # Raise error for unsupported input types
    raise ValueError(
        f"Input must be of type MultiModalInput, (user_prompt[str], system_prompt[str]), "
        f"dict with 'user_prompt' and optional 'system_prompt', or user_prompt[str]. "
        f"Received type: {type(llm_input)}"
    )
