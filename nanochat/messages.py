"""
Nanochat Structured Message Format

This module defines the message schema for Nanochat chat interactions.
Messages are validated and structured according to the token protocol.

Design:
    - Type-safe message structure with validation
    - Support for reasoning, tool calls, and multimodal content
    - Serialization to/from dictionaries for dataset storage
    - Protocol compliance checking
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any, Literal
import json

from nanochat.token_protocol import (
    Role, ReasoningLevel,
    VALID_ROLES, VALID_REASONING_LEVELS,
    validate_role, validate_reasoning_level
)


@dataclass
class ToolCall:
    """
    Represents a tool invocation by the assistant.
    
    Attributes:
        id: Unique identifier for this tool call
        name: Name of the tool to invoke
        arguments: Dictionary of arguments to pass to the tool
        
    Example:
        ToolCall(
            id="call_001",
            name="get_weather",
            arguments={"location": "San Francisco", "units": "celsius"}
        )
    """
    id: str
    name: str
    arguments: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        """Create from dictionary"""
        return cls(
            id=data["id"],
            name=data["name"],
            arguments=data["arguments"],
        )
    
    def to_json(self) -> str:
        """Convert to JSON string for inclusion in token sequence"""
        return json.dumps(self.to_dict())


@dataclass
class Content:
    """
    Multi-modal content item.
    
    Currently supports:
        - text: Plain text content
        - image: Image placeholder (future)
        - video: Video placeholder (future)
        - audio: Audio placeholder (future)
        
    For multimodal content, the actual media data is stored separately
    and the token sequence contains a placeholder token (<|image|>, etc.).
    The model processor handles embedding the media.
    
    Example:
        # Text content
        Content(type="text", text="Hello world")
        
        # Image placeholder (future)
        Content(type="image", image=PIL.Image.open("photo.jpg"))
    """
    type: Literal["text", "image", "video", "audio"]
    
    # For text
    text: Optional[str] = None
    
    # For media (future - placeholders for now)
    image: Optional[Any] = None
    video: Optional[Any] = None
    audio: Optional[Any] = None
    
    def __post_init__(self):
        """Validate content structure"""
        if self.type == "text" and self.text is None:
            raise ValueError("text content requires 'text' field")
        # Future: validate media types when implemented
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {"type": self.type}
        if self.type == "text":
            result["text"] = self.text
        # Future: handle media serialization
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Content":
        """Create from dictionary"""
        content_type = data["type"]
        if content_type == "text":
            return cls(type="text", text=data["text"])
        # Future: handle other content types
        raise ValueError(f"Unsupported content type: {content_type}")


@dataclass
class Message:
    """
    Structured message following Nanochat protocol v1.0.0
    
    All messages must have a role and content. Additional fields depend on role:
    
    - assistant: Can have reasoning, reasoning_level, tool_calls
    - tool: Must have name and tool_call_id
    - Other roles: Only role and content
    
    The reasoning structure is ALWAYS present for assistant messages:
    - If reasoning_level="none" or not specified: <|think_none|><|think_end|>
    - If reasoning_level specified: <|think_X|>reasoning...<|think_end|>
    
    Examples:
        # Simple user message
        Message(role="user", content="What is 2+2?")
        
        # Assistant with reasoning
        Message(
            role="assistant",
            reasoning="This is basic arithmetic. 2+2 equals 4.",
            reasoning_level="low",
            content="The answer is 4."
        )
        
        # Assistant without explicit reasoning (think_none)
        Message(
            role="assistant",
            reasoning_level="none",
            content="The answer is 4."
        )
        
        # Assistant with tool call
        Message(
            role="assistant",
            reasoning="I need to check the current time.",
            reasoning_level="medium",
            tool_calls=[
                ToolCall(id="call_1", name="get_time", arguments={})
            ],
            content=""  # Content can be empty when only tool calls
        )
        
        # Tool result
        Message(
            role="tool",
            name="get_time",
            tool_call_id="call_1",
            content="2026-09-18 14:30:00 UTC"
        )
        
        # System message
        Message(
            role="system",
            content="You are a helpful AI assistant."
        )
        
        # Developer message (for tool schemas, constraints, etc.)
        Message(
            role="developer",
            content="Available tools: get_time, get_weather"
        )
    """
    role: Role
    content: Union[str, List[Content]] = ""
    
    # Reasoning (assistant only)
    reasoning: Optional[str] = None
    reasoning_level: Optional[ReasoningLevel] = None
    
    # Tool calls (assistant only)
    tool_calls: Optional[List[ToolCall]] = None
    
    # Tool result metadata (tool role only)
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate message structure according to protocol"""
        # Validate role
        validate_role(self.role)
        
        # Validate reasoning level if provided
        if self.reasoning_level is not None:
            validate_reasoning_level(self.reasoning_level)
            
            # Reasoning only valid for assistant
            if self.role != "assistant":
                raise ValueError(
                    f"reasoning_level only valid for assistant role, "
                    f"got role='{self.role}'"
                )
        
        # Validate tool calls
        if self.tool_calls is not None:
            if self.role != "assistant":
                raise ValueError(
                    f"tool_calls only valid for assistant role, "
                    f"got role='{self.role}'"
                )
        
        # Validate tool role requirements
        if self.role == "tool":
            if self.name is None or self.tool_call_id is None:
                raise ValueError(
                    "tool role requires 'name' and 'tool_call_id' fields"
                )
        
        # Convert string content to Content object for consistency
        if isinstance(self.content, str) and self.content:
            # Keep as string for simplicity - will handle in template
            pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary (for serialization/dataset storage).
        
        Returns:
            Dictionary representation of the message
        """
        result = {"role": self.role}
        
        # Handle content
        if isinstance(self.content, str):
            result["content"] = self.content
        else:
            result["content"] = [c.to_dict() for c in self.content]
        
        # Add optional fields
        if self.reasoning is not None:
            result["reasoning"] = self.reasoning
        if self.reasoning_level is not None:
            result["reasoning_level"] = self.reasoning_level
        if self.tool_calls is not None:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.name is not None:
            result["name"] = self.name
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation of message
            
        Returns:
            Message instance
        """
        # Handle content
        content = data.get("content", "")
        if isinstance(content, list):
            content = [Content.from_dict(item) for item in content]
        
        # Handle tool calls
        tool_calls = None
        if "tool_calls" in data and data["tool_calls"] is not None:
            tool_calls = [ToolCall.from_dict(tc) for tc in data["tool_calls"]]
        
        return cls(
            role=data["role"],
            content=content,
            reasoning=data.get("reasoning"),
            reasoning_level=data.get("reasoning_level"),
            tool_calls=tool_calls,
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
        )
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Message":
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))
    
    def is_assistant_message(self) -> bool:
        """Check if this is an assistant message"""
        return self.role == "assistant"
    
    def has_reasoning(self) -> bool:
        """Check if this message has explicit reasoning content"""
        return self.reasoning is not None and len(self.reasoning) > 0
    
    def has_tool_calls(self) -> bool:
        """Check if this message has tool calls"""
        return self.tool_calls is not None and len(self.tool_calls) > 0


def create_conversation(
    system_message: Optional[str] = None,
    developer_message: Optional[str] = None,
    messages: Optional[List[Message]] = None
) -> List[Message]:
    """
    Create a conversation with optional system/developer messages.
    
    Args:
        system_message: Optional system message content
        developer_message: Optional developer message content
        messages: List of user/assistant messages
        
    Returns:
        Complete conversation as list of Messages
        
    Example:
        conv = create_conversation(
            system_message="You are a helpful assistant.",
            developer_message="Use tools when needed.",
            messages=[
                Message(role="user", content="What time is it?"),
                Message(
                    role="assistant",
                    reasoning="Need to check current time",
                    reasoning_level="low",
                    tool_calls=[ToolCall(id="1", name="get_time", arguments={})]
                ),
            ]
        )
    """
    conversation = []
    
    if system_message:
        conversation.append(Message(role="system", content=system_message))
    
    if developer_message:
        conversation.append(Message(role="developer", content=developer_message))
    
    if messages:
        conversation.extend(messages)
    
    return conversation


if __name__ == "__main__":
    # Test message creation and validation
    print("Nanochat Message Schema Tests")
    print("=" * 50)
    
    # Test 1: Simple user message
    msg1 = Message(role="user", content="Hello!")
    print("✓ User message:", msg1.to_dict())
    
    # Test 2: Assistant with reasoning
    msg2 = Message(
        role="assistant",
        reasoning="Simple greeting response",
        reasoning_level="low",
        content="Hello! How can I help?"
    )
    print("✓ Assistant with reasoning:", msg2.to_dict())
    
    # Test 3: Assistant with tool call
    msg3 = Message(
        role="assistant",
        reasoning="Need current time",
        reasoning_level="medium",
        tool_calls=[
            ToolCall(id="call_1", name="get_time", arguments={})
        ],
        content=""
    )
    print("✓ Assistant with tool call:", msg3.to_dict())
    
    # Test 4: Tool result
    msg4 = Message(
        role="tool",
        name="get_time",
        tool_call_id="call_1",
        content="14:30:00"
    )
    print("✓ Tool result:", msg4.to_dict())
    
    # Test 5: Invalid role (should raise)
    try:
        Message(role="invalid", content="test")
        print("✗ Should have raised ValueError for invalid role")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    # Test 6: Invalid reasoning level
    try:
        Message(role="assistant", reasoning_level="invalid", content="test")
        print("✗ Should have raised ValueError for invalid reasoning_level")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    # Test 7: Reasoning on non-assistant
    try:
        Message(role="user", reasoning_level="high", content="test")
        print("✗ Should have raised ValueError for reasoning on non-assistant")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    print("\n✓ All validation tests passed!")
