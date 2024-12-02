import asyncio
from inspect import iscoroutinefunction
from typing import Any

from pydantic import BaseModel

from ..schema import MultiModalInput


class BaseModifier:
    """Base class for modifiers that can intercept and modify reasoner execution"""

    def __init__(self):
        """Verify that instance implements the required method"""
        if self.__class__.modify == BaseModifier.modify:
            raise TypeError(
                f"Can't instantiate class {self.__class__.__name__} with abstract method 'modify'"
            )

    def modify(
        self, input: MultiModalInput, schema: BaseModel, model: Any
    ) -> BaseModel:
        """Synchronous modify implementation that subclasses must implement"""
        raise NotImplementedError("Subclasses must implement modify()")

    async def async_modify(
        self, input: MultiModalInput, schema: BaseModel, model: Any
    ) -> BaseModel:
        """
        Default async implementation that calls sync modify.
        Subclasses can override this for true async implementation.
        """
        # Check if the subclass's modify is actually async
        if iscoroutinefunction(self.__class__.modify):
            # If modify is async, await it
            return await self.modify(input, schema, model)
        else:
            # If modify is sync, run it in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.modify, input, schema, model)

    def __reduce__(self):
        """Make the class pickleable by specifying how to reconstruct it"""
        return (self.__class__, (), self.__dict__)
