from pydantic import BaseModel


class llmMessage(BaseModel):
    """
    Represents a message in an LLM (large language model) conversation.

    Attributes:
        role (str): The role of the message sender. Defaults to "user".
        content (str): The content of the message. Defaults to an empty string.
    """

    role: str = "user"
    content: str = ""

    def to_dict(self):
        return self.content
