"""
Pydantic output parser — drop-in replacement for langchain_core.output_parsers.PydanticOutputParser.

Usage:
    from ai_router.parsers import PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=MySchema)
    instructions = parser.get_format_instructions()
    result = parser.parse(llm_response_text)
"""

import json
import re
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class PydanticOutputParser:
    """Parse LLM text output into a Pydantic model."""

    def __init__(self, pydantic_object: Type[T]):
        self.pydantic_object = pydantic_object

    def get_format_instructions(self) -> str:
        schema = self.pydantic_object.model_json_schema()
        return (
            "The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```\n\n"
            "Return ONLY the JSON object, no additional text."
        )

    def parse(self, text: str) -> T:
        """Extract JSON from text and parse into the Pydantic model."""
        json_str = self._extract_json(text)
        data = json.loads(json_str)
        return self.pydantic_object.model_validate(data)

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON object from text, handling markdown code fences."""
        # Try markdown code fence first
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try to find a JSON object directly
        # Find the first { and last } to extract the outermost JSON object
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return text[first_brace : last_brace + 1]

        # Return as-is and let json.loads handle the error
        return text.strip()
