from typing import List
from pydantic import BaseModel, Field


class Source(BaseModel):
    """A source document."""

    url: str = Field(description="The URL of the source document.")


class AgentResponse(BaseModel):
    """The answer with sources."""

    answer: str = Field(
        description="The agent answer to the query."
    )  ## ... is ellipsis which means this is a required field
    sources: List[Source] = Field(
        default_factory=list,
        description="List of source documents used to generate the answer.",  ## default_factory=list means if no value is provided, it will be an empty list
    )
