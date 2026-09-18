# Nanochat Token Protocol & Chat Architecture Implementation Plan

## Overview

This plan outlines the implementation of a robust, extensible tokenizer and chat protocol architecture for Nanochat. The goal is to create a foundation that supports reasoning, tool use, and future multimodal capabilities while maintaining Nanochat's compact vocabulary philosophy.

## Core Principles

1. **Small vocabulary ≠ limited capability**: Keep 32K-64K lexical tokens + 256 reserved control tokens
2. **Single source of truth**: Chat template controls all formatting (training + inference)
3. **Structural consistency**: Reasoning blocks always present in assistant messages
4. **Future-proof**: Reserved token namespace for expansion without breaking changes
5. **HuggingFace compatible**: Export standard tokenizer artifacts

---

## Phase 1: Token Architecture & Tokenizer Core

### 1.1 Define Token Namespace

**File**: `nanochat/token_protocol.py` (new)

```python
"""
Nanochat Token Protocol Definition
Version: 1.0.0
"""

from enum import IntEnum
from typing import Literal, List, Optional
from dataclasses import dataclass

# Token ranges
LEXICAL_VOCAB_SIZE = 32000  # Base BPE vocabulary (0-31999)
CONTROL_TOKEN_BASE = 32000
CONTROL_TOKEN_LIMIT = 32256
TOTAL_VOCAB_SIZE = 32256

class ControlTokenID(IntEnum):
    """Control token IDs - never hardcode these elsewhere in codebase"""
    # Core tokens (32000-32009)
    PAD = 32000
    BOS = 32001
    EOS = 32002
    UNK = 32003
    
    # Message structure (32010-32019)
    MESSAGE_START = 32010
    MESSAGE_END = 32011
    
    # Roles (32020-32029)
    SYSTEM = 32020
    DEVELOPER = 32021
    USER = 32022
    ASSISTANT = 32023
    TOOL = 32024
    
    # Reasoning (32030-32039)
    THINK_NONE = 32030
    THINK_LOW = 32031
    THINK_MEDIUM = 32032
    THINK_HIGH = 32033
    THINK_END = 32034
    
    # Tool calls (32040-32049)
    TOOL_CALL = 32040
    TOOL_CALL_END = 32041
    TOOL_RESULT = 32042
    TOOL_RESULT_END = 32043
    
    # Multimodal (reserved for future, 32050-32059)
    IMAGE = 32050
    IMAGE_END = 32051
    VIDEO = 32052
    VIDEO_END = 32053
    AUDIO = 32054
    AUDIO_END = 32055
    
    # Reserved for future expansion (32060-32255)
    # These exist in vocab but have no semantic meaning yet

# Token strings
SPECIAL_TOKENS = {
    # Core
    "pad": "<|pad|>",
    "bos": "<|bos|>",
    "eos": "<|eos|>",
    "unk": "<|unk|>",
    
    # Message structure
    "message_start": "<|message_start|>",
    "message_end": "<|message_end|>",
    
    # Roles
    "system": "<|system|>",
    "developer": "<|developer|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool": "<|tool|>",
    
    # Reasoning
    "think_none": "<|think_none|>",
    "think_low": "<|think_low|>",
    "think_medium": "<|think_medium|>",
    "think_high": "<|think_high|>",
    "think_end": "<|think_end|>",
    
    # Tool calls
    "tool_call": "<|tool_call|>",
    "tool_call_end": "<|tool_call_end|>",
    "tool_result": "<|tool_result|>",
    "tool_result_end": "<|tool_result_end|>",
    
    # Multimodal (reserved)
    "image": "<|image|>",
    "image_end": "<|image_end|>",
    "video": "<|video|>",
    "video_end": "<|video_end|>",
    "audio": "<|audio|>",
    "audio_end": "<|audio_end|>",
}

# Create reserved token placeholders
RESERVED_TOKENS = {
    f"reserved_{i:03d}": f"<|reserved_{i:03d}|>"
    for i in range(60, 256)  # 32060-32255
}

# Valid enums for message validation
Role = Literal["system", "developer", "user", "assistant", "tool"]
ReasoningLevel = Literal["none", "low", "medium", "high"]

VALID_ROLES = {"system", "developer", "user", "assistant", "tool"}
VALID_REASONING_LEVELS = {"none", "low", "medium", "high"}

# Protocol version
PROTOCOL_VERSION = "1.0.0"
```

**Success criteria**: 
- ✓ Token namespace clearly defined
- ✓ No magic numbers scattered in code
- ✓ Easy to extend without breaking changes

---

### 1.2 Update Tokenizer Training

**File**: `scripts/tok_train.py` (modify)

**Changes needed**:
1. Train base BPE with vocab size 32000 (not 32768)
2. After training, append special tokens in defined order
3. Add reserved tokens to vocabulary
4. Save tokenizer with all tokens included

```python
# In tok_train.py, after BPE training:

from nanochat.token_protocol import (
    SPECIAL_TOKENS, RESERVED_TOKENS, 
    TOTAL_VOCAB_SIZE, CONTROL_TOKEN_BASE
)

# Add special tokens in defined order
special_token_list = []
for token_id in sorted(ControlTokenID):
    token_name = token_id.name.lower()
    if token_name in SPECIAL_TOKENS:
        special_token_list.append(SPECIAL_TOKENS[token_name])

# Add reserved tokens
for i in range(60, 256):
    special_token_list.append(RESERVED_TOKENS[f"reserved_{i:03d}"])

# Add to tokenizer
tokenizer.add_special_tokens(special_token_list)

assert len(tokenizer) == TOTAL_VOCAB_SIZE, \
    f"Expected {TOTAL_VOCAB_SIZE} tokens, got {len(tokenizer)}"
```

**Success criteria**:
- ✓ Tokenizer has exactly 32256 tokens
- ✓ Special tokens at correct IDs
- ✓ Reserved tokens exist but unused

---

### 1.3 Create Message Schema

**File**: `nanochat/messages.py` (new)

```python
"""
Nanochat structured message format
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
from nanochat.token_protocol import (
    Role, ReasoningLevel,
    VALID_ROLES, VALID_REASONING_LEVELS
)

@dataclass
class ToolCall:
    """Represents a tool invocation"""
    id: str
    name: str
    arguments: Dict[str, Any]

@dataclass
class Content:
    """Multi-modal content item"""
    type: Literal["text", "image", "video", "audio"]
    
    # For text
    text: Optional[str] = None
    
    # For media (future)
    image: Optional[Any] = None
    video: Optional[Any] = None
    audio: Optional[Any] = None
    
    def __post_init__(self):
        if self.type == "text" and self.text is None:
            raise ValueError("text content requires 'text' field")

@dataclass
class Message:
    """
    Structured message following Nanochat protocol v1.0.0
    
    Examples:
        # Simple user message
        Message(role="user", content="Hello!")
        
        # Assistant with reasoning
        Message(
            role="assistant",
            reasoning="User is greeting me, I should respond politely.",
            reasoning_level="low",
            content="Hello! How can I help you?"
        )
        
        # Assistant with tool call
        Message(
            role="assistant",
            reasoning="Need to check current time.",
            reasoning_level="medium",
            tool_calls=[
                ToolCall(id="call_1", name="get_time", arguments={})
            ]
        )
        
        # Tool result
        Message(
            role="tool",
            name="get_time",
            tool_call_id="call_1",
            content="2026-09-18 14:30:00"
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
        """Validate message structure"""
        # Validate role
        if self.role not in VALID_ROLES:
            raise ValueError(
                f"Invalid role '{self.role}'. "
                f"Must be one of: {VALID_ROLES}"
            )
        
        # Validate reasoning level
        if self.reasoning_level is not None:
            if self.reasoning_level not in VALID_REASONING_LEVELS:
                raise ValueError(
                    f"Invalid reasoning_level '{self.reasoning_level}'. "
                    f"Must be one of: {VALID_REASONING_LEVELS}"
                )
            
            # Reasoning only valid for assistant
            if self.role != "assistant":
                raise ValueError(
                    f"reasoning_level only valid for assistant role, "
                    f"got role='{self.role}'"
                )
        
        # Validate tool fields
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError(
                f"tool_calls only valid for assistant role, "
                f"got role='{self.role}'"
            )
        
        if self.role == "tool":
            if self.name is None or self.tool_call_id is None:
                raise ValueError(
                    "tool role requires 'name' and 'tool_call_id' fields"
                )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)"""
        result = {"role": self.role}
        
        if isinstance(self.content, str):
            result["content"] = self.content
        else:
            result["content"] = [
                {"type": c.type, **{c.type: getattr(c, c.type)}}
                for c in self.content
            ]
        
        if self.reasoning is not None:
            result["reasoning"] = self.reasoning
        if self.reasoning_level is not None:
            result["reasoning_level"] = self.reasoning_level
        if self.tool_calls is not None:
            result["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in self.tool_calls
            ]
        if self.name is not None:
            result["name"] = self.name
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary"""
        # Handle content
        content = data.get("content", "")
        if isinstance(content, list):
            content = [
                Content(type=item["type"], **{item["type"]: item[item["type"]]})
                for item in content
            ]
        
        # Handle tool calls
        tool_calls = None
        if "tool_calls" in data:
            tool_calls = [
                ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
                for tc in data["tool_calls"]
            ]
        
        return cls(
            role=data["role"],
            content=content,
            reasoning=data.get("reasoning"),
            reasoning_level=data.get("reasoning_level"),
            tool_calls=tool_calls,
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
        )
```

**Success criteria**:
- ✓ Type-safe message structure
- ✓ Validation prevents invalid messages
- ✓ Supports all required features (reasoning, tools, multimodal placeholders)

---

## Phase 2: Chat Template

### 2.1 Create Jinja Template

**File**: `nanochat/chat_template.jinja` (new)

```jinja2
{#
Nanochat Chat Template v1.0.0

This template converts structured messages into the token sequence
the model is trained on. Training and inference MUST use this template.

Message structure:
- All messages wrapped in <|message_start|> ... <|message_end|>
- Role marker follows message_start
- Assistant messages always have reasoning block
- Tool calls are JSON-serialized

Reasoning behavior:
- reasoning_level="none": <|think_none|><|think_end|> (immediate)
- reasoning_level="low/medium/high": <|think_X|> ... <|think_end|>
#}

{%- for message in messages %}
{{- '<|message_start|>' -}}
{{- '<|' + message.role + '|>' -}}

{# Assistant messages always have reasoning structure #}
{%- if message.role == 'assistant' %}
    {%- if message.reasoning_level == 'none' or (not message.reasoning and not message.reasoning_level) %}
        {{- '<|think_none|><|think_end|>' -}}
    {%- elif message.reasoning_level == 'low' %}
        {{- '<|think_low|>' -}}
        {{- message.reasoning if message.reasoning else '' -}}
        {{- '<|think_end|>' -}}
    {%- elif message.reasoning_level == 'medium' %}
        {{- '<|think_medium|>' -}}
        {{- message.reasoning if message.reasoning else '' -}}
        {{- '<|think_end|>' -}}
    {%- elif message.reasoning_level == 'high' %}
        {{- '<|think_high|>' -}}
        {{- message.reasoning if message.reasoning else '' -}}
        {{- '<|think_end|>' -}}
    {%- else %}
        {{- raise_exception("Invalid reasoning_level: " + (message.reasoning_level or "null")) -}}
    {%- endif %}
{%- endif %}

{# Content (text or multimodal) #}
{%- if message.content is string %}
    {{- message.content -}}
{%- else %}
    {%- for item in message.content %}
        {%- if item.type == 'text' %}
            {{- item.text -}}
        {%- elif item.type == 'image' %}
            {{- '<|image|>' -}}
        {%- elif item.type == 'video' %}
            {{- '<|video|>' -}}
        {%- elif item.type == 'audio' %}
            {{- '<|audio|>' -}}
        {%- endif %}
    {%- endfor %}
{%- endif %}

{# Tool calls (assistant messages only) #}
{%- if message.tool_calls %}
    {%- for call in message.tool_calls %}
        {{- '<|tool_call|>' -}}
        {{- call | tojson -}}
        {{- '<|tool_call_end|>' -}}
    {%- endfor %}
{%- endif %}

{{- '<|message_end|>' -}}
{%- endfor %}

{# Generation prompt #}
{%- if add_generation_prompt %}
    {{- '<|message_start|><|assistant|>' -}}
    {%- if reasoning_level == 'none' %}
        {{- '<|think_none|><|think_end|>' -}}
    {%- elif reasoning_level == 'low' %}
        {{- '<|think_low|>' -}}
    {%- elif reasoning_level == 'medium' %}
        {{- '<|think_medium|>' -}}
    {%- elif reasoning_level == 'high' %}
        {{- '<|think_high|>' -}}
    {%- else %}
        {{- raise_exception("Invalid generation reasoning_level: " + (reasoning_level or "none")) -}}
    {%- endif %}
{%- endif %}
```

**Success criteria**:
- ✓ Consistent formatting for all message types
- ✓ Reasoning structure always present for assistant
- ✓ Tool calls properly formatted
- ✓ Generation prompt behavior correct

---

### 2.2 Implement apply_chat_template

**File**: `nanochat/tokenizer.py` (modify)

```python
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from typing import List, Union, Dict, Any
from nanochat.messages import Message

class NanochatTokenizer:
    """Enhanced tokenizer with chat template support"""
    
    def __init__(self, tokenizer_path: str):
        # Load base tokenizer (existing code)
        self.tokenizer = load_base_tokenizer(tokenizer_path)
        
        # Load chat template
        template_path = Path(__file__).parent / "chat_template.jinja"
        env = Environment(loader=FileSystemLoader(template_path.parent))
        self.chat_template = env.get_template(template_path.name)
        
        # Token ID lookups
        from nanochat.token_protocol import SPECIAL_TOKENS
        self.special_token_ids = {
            name: self.tokenizer.encode(token, add_special_tokens=False)[0]
            for name, token in SPECIAL_TOKENS.items()
        }
    
    def apply_chat_template(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        reasoning_level: str = "medium",
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        return_dict: bool = False,
    ):
        """
        Convert structured messages to tokens using chat template.
        
        This is the SINGLE SOURCE OF TRUTH for message formatting.
        All training and inference must use this method.
        
        Args:
            messages: List of Message objects or dicts
            reasoning_level: Reasoning level for generation ("none", "low", "medium", "high")
            add_generation_prompt: Add assistant prompt for generation
            tokenize: Return token IDs (True) or string (False)
            return_dict: Return dict with additional info
        
        Returns:
            Token IDs (list), string, or dict depending on parameters
        """
        # Convert dicts to Message objects
        if messages and isinstance(messages[0], dict):
            messages = [Message.from_dict(m) for m in messages]
        
        # Validate reasoning level
        from nanochat.token_protocol import VALID_REASONING_LEVELS
        if reasoning_level not in VALID_REASONING_LEVELS:
            raise ValueError(
                f"Invalid reasoning_level '{reasoning_level}'. "
                f"Must be one of: {VALID_REASONING_LEVELS}"
            )
        
        # Render template
        formatted_text = self.chat_template.render(
            messages=[m.to_dict() for m in messages],
            reasoning_level=reasoning_level,
            add_generation_prompt=add_generation_prompt,
        )
        
        if not tokenize:
            return formatted_text
        
        # Tokenize
        token_ids = self.tokenizer.encode(formatted_text, add_special_tokens=False)
        
        if return_dict:
            return {
                "input_ids": token_ids,
                "formatted_text": formatted_text,
                "reasoning_level": reasoning_level,
            }
        
        return token_ids
    
    def get_special_token_id(self, name: str) -> int:
        """Get ID for a special token by name"""
        if name not in self.special_token_ids:
            raise ValueError(f"Unknown special token: {name}")
        return self.special_token_ids[name]
```

**Success criteria**:
- ✓ Single method for all formatting
- ✓ Works with Message objects or dicts
- ✓ Validates reasoning levels
- ✓ Returns tokens or text as needed

---

## Phase 3: Training Integration

### 3.1 Update Dataset Formatting

**File**: `nanochat/dataset.py` (modify)

**Remove all manual formatting code**. Replace with:

```python
def format_conversation_for_training(
    messages: List[Message],
    tokenizer: NanochatTokenizer,
    reasoning_level: str = "medium",
) -> Dict[str, Any]:
    """
    Format a conversation for training using chat template.
    
    This replaces all manual formatting in the codebase.
    """
    # Use tokenizer's chat template
    result = tokenizer.apply_chat_template(
        messages=messages,
        reasoning_level=reasoning_level,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
    )
    
    # Create loss mask (mask input, train on assistant output)
    token_ids = result["input_ids"]
    loss_mask = create_loss_mask(token_ids, tokenizer)
    
    return {
        "input_ids": token_ids,
        "loss_mask": loss_mask,
        "reasoning_level": reasoning_level,
    }

def create_loss_mask(token_ids: List[int], tokenizer: NanochatTokenizer) -> List[bool]:
    """
    Create loss mask for training.
    
    Rules:
    - Mask: system, developer, user, tool messages
    - Train: assistant reasoning + content
    - Special handling for reasoning tokens based on mode
    """
    mask = []
    in_assistant = False
    in_reasoning = False
    
    assistant_id = tokenizer.get_special_token_id("assistant")
    think_none_id = tokenizer.get_special_token_id("think_none")
    think_end_id = tokenizer.get_special_token_id("think_end")
    message_end_id = tokenizer.get_special_token_id("message_end")
    
    for token_id in token_ids:
        if token_id == assistant_id:
            in_assistant = True
            mask.append(False)  # Mask role token itself
        elif in_assistant:
            if token_id == think_none_id:
                # Mask <|think_none|> and following <|think_end|>
                in_reasoning = "none"
                mask.append(False)
            elif token_id in [
                tokenizer.get_special_token_id("think_low"),
                tokenizer.get_special_token_id("think_medium"),
                tokenizer.get_special_token_id("think_high"),
            ]:
                # Train on reasoning content
                in_reasoning = True
                mask.append(True)
            elif token_id == think_end_id:
                mask.append(True if in_reasoning == True else False)
                in_reasoning = False
            elif token_id == message_end_id:
                in_assistant = False
                mask.append(False)
            else:
                # Train on assistant content (reasoning or final answer)
                mask.append(True)
        else:
            # Mask all non-assistant content
            mask.append(False)
    
    return mask
```

**Files to update**:
- `scripts/chat_sft.py`: Use `apply_chat_template` instead of manual formatting
- `scripts/chat_rl.py`: Same
- `scripts/chat_eval.py`: Same
- Any other scripts that format conversations

**Success criteria**:
- ✓ No manual formatting anywhere in codebase
- ✓ Training uses exact same format as inference
- ✓ Loss masking respects reasoning structure

---

### 3.2 Update Inference/Generation

**File**: `scripts/chat_cli.py` (modify)

```python
def generate_response(
    model,
    tokenizer: NanochatTokenizer,
    messages: List[Message],
    reasoning_level: str = "medium",
    max_new_tokens: int = 512,
) -> Message:
    """
    Generate assistant response with reasoning.
    
    Uses same chat template as training.
    """
    # Format with generation prompt
    input_ids = tokenizer.apply_chat_template(
        messages=messages,
        reasoning_level=reasoning_level,
        add_generation_prompt=True,
        tokenize=True,
    )
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            torch.tensor([input_ids]).to(model.device),
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.get_special_token_id("message_end"),
        )
    
    # Decode only new tokens
    new_tokens = output_ids[0][len(input_ids):]
    generated_text = tokenizer.decode(new_tokens)
    
    # Parse response
    response = parse_assistant_message(generated_text, tokenizer)
    return response

def parse_assistant_message(text: str, tokenizer: NanochatTokenizer) -> Message:
    """Parse generated assistant message into structured format"""
    # Extract reasoning block
    reasoning = None
    reasoning_level = None
    content = text
    
    # Pattern: <|think_X|>...<|think_end|>content<|message_end|>
    import re
    
    for level in ["none", "low", "medium", "high"]:
        pattern = f"<\\|think_{level}\\|>(.*?)<\\|think_end\\|>(.*?)<\\|message_end\\|>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            if level != "none":
                reasoning = match.group(1).strip()
            reasoning_level = level
            content = match.group(2).strip()
            break
    
    # Parse tool calls if present
    tool_calls = None
    if "<|tool_call|>" in content:
        tool_calls = []
        for match in re.finditer(
            r"<\|tool_call\|>(.*?)<\|tool_call_end\|>",
            content,
            re.DOTALL
        ):
            call_json = json.loads(match.group(1))
            tool_calls.append(ToolCall(**call_json))
        
        # Remove tool call markup from content
        content = re.sub(
            r"<\|tool_call\|>.*?<\|tool_call_end\|>",
            "",
            content,
            flags=re.DOTALL
        ).strip()
    
    return Message(
        role="assistant",
        reasoning=reasoning,
        reasoning_level=reasoning_level,
        content=content,
        tool_calls=tool_calls,
    )
```

**Success criteria**:
- ✓ Generation uses same template as training
- ✓ Can control reasoning level at inference time
- ✓ Response properly parsed into Message structure

---

## Phase 4: HuggingFace Compatibility

### 4.1 Export Tokenizer Artifacts

**File**: `scripts/export_tokenizer_hf.py` (new)

```python
"""
Export Nanochat tokenizer in HuggingFace format.

Creates:
- tokenizer.json (vocab + merges)
- tokenizer_config.json (config)
- special_tokens_map.json (special tokens)
- chat_template.jinja (template)
"""

from pathlib import Path
import json
from nanochat.tokenizer import NanochatTokenizer
from nanochat.token_protocol import (
    SPECIAL_TOKENS, PROTOCOL_VERSION, TOTAL_VOCAB_SIZE
)

def export_tokenizer_hf(
    nanochat_tokenizer_path: str,
    output_dir: str,
):
    """Export tokenizer in HF format"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer = NanochatTokenizer(nanochat_tokenizer_path)
    
    # 1. Copy tokenizer.json (base vocabulary)
    # This is the BPE vocab + merges
    shutil.copy(
        Path(nanochat_tokenizer_path) / "tokenizer.json",
        output_dir / "tokenizer.json"
    )
    
    # 2. tokenizer_config.json
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 2048,
        "padding_side": "left",
        "truncation_side": "right",
        "vocab_size": TOTAL_VOCAB_SIZE,
        "bos_token": SPECIAL_TOKENS["bos"],
        "eos_token": SPECIAL_TOKENS["eos"],
        "unk_token": SPECIAL_TOKENS["unk"],
        "pad_token": SPECIAL_TOKENS["pad"],
        "add_bos_token": True,
        "add_eos_token": False,
        "nanochat_protocol_version": PROTOCOL_VERSION,
        "chat_template_file": "chat_template.jinja",
    }
    
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # 3. special_tokens_map.json
    special_tokens_map = {
        "bos_token": SPECIAL_TOKENS["bos"],
        "eos_token": SPECIAL_TOKENS["eos"],
        "unk_token": SPECIAL_TOKENS["unk"],
        "pad_token": SPECIAL_TOKENS["pad"],
        "additional_special_tokens": [
            token for name, token in SPECIAL_TOKENS.items()
            if name not in ["bos", "eos", "unk", "pad"]
        ],
    }
    
    with open(output_dir / "special_tokens_map.json", "w") as f:
        json.dump(special_tokens_map, f, indent=2)
    
    # 4. Copy chat template
    shutil.copy(
        Path(__file__).parent.parent / "nanochat" / "chat_template.jinja",
        output_dir / "chat_template.jinja"
    )
    
    print(f"✓ Exported HF tokenizer to {output_dir}")
    print(f"  - Protocol version: {PROTOCOL_VERSION}")
    print(f"  - Vocab size: {TOTAL_VOCAB_SIZE}")
    print(f"  - Special tokens: {len(SPECIAL_TOKENS)}")
```

**Success criteria**:
- ✓ Can load with `AutoTokenizer.from_pretrained()`
- ✓ `tokenizer.apply_chat_template()` works
- ✓ Compatible with HF ecosystem

---

## Phase 5: Testing & Validation

### 5.1 Unit Tests

**File**: `tests/test_token_protocol.py` (new)

```python
"""Test token protocol and chat template"""

import pytest
from nanochat.messages import Message, ToolCall
from nanochat.tokenizer import NanochatTokenizer
from nanochat.token_protocol import TOTAL_VOCAB_SIZE

def test_vocab_size():
    """Tokenizer has correct total size"""
    tokenizer = NanochatTokenizer("path/to/tokenizer")
    assert len(tokenizer) == TOTAL_VOCAB_SIZE

def test_special_token_ids():
    """Special tokens at correct IDs"""
    tokenizer = NanochatTokenizer("path/to/tokenizer")
    
    # Check a few key tokens
    assert tokenizer.get_special_token_id("bos") == 32001
    assert tokenizer.get_special_token_id("assistant") == 32023
    assert tokenizer.get_special_token_id("think_medium") == 32032

def test_message_validation():
    """Message validation works"""
    # Valid message
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    
    # Invalid role
    with pytest.raises(ValueError, match="Invalid role"):
        Message(role="invalid", content="test")
    
    # Invalid reasoning level
    with pytest.raises(ValueError, match="Invalid reasoning_level"):
        Message(
            role="assistant",
            content="test",
            reasoning_level="invalid"
        )
    
    # Reasoning on non-assistant
    with pytest.raises(ValueError, match="only valid for assistant"):
        Message(
            role="user",
            content="test",
            reasoning_level="medium"
        )

def test_chat_template_basic():
    """Chat template formats correctly"""
    tokenizer = NanochatTokenizer("path/to/tokenizer")
    
    messages = [
        Message(role="user", content="What is 2+2?"),
        Message(
            role="assistant",
            reasoning="Simple arithmetic",
            reasoning_level="low",
            content="4"
        ),
    ]
    
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    
    # Check structure
    assert "<|message_start|>" in formatted
    assert "<|user|>" in formatted
    assert "<|assistant|>" in formatted
    assert "<|think_low|>" in formatted
    assert "Simple arithmetic" in formatted
    assert "<|think_end|>" in formatted
    assert "4" in formatted
    assert "<|message_end|>" in formatted

def test_reasoning_none():
    """think_none format is correct"""
    tokenizer = NanochatTokenizer("path/to/tokenizer")
    
    messages = [
        Message(
            role="assistant",
            reasoning_level="none",
            content="Quick answer"
        ),
    ]
    
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    
    assert "<|think_none|><|think_end|>" in formatted
    assert "Quick answer" in formatted

def test_generation_prompt():
    """Generation prompt formats correctly"""
    tokenizer = NanochatTokenizer("path/to/tokenizer")
    
    messages = [
        Message(role="user", content="Hello"),
    ]
    
    # With reasoning
    formatted = tokenizer.apply_chat_template(
        messages,
        reasoning_level="high",
        add_generation_prompt=True,
        tokenize=False
    )
    
    assert formatted.endswith("<|message_start|><|assistant|><|think_high|>")
    
    # Without reasoning
    formatted = tokenizer.apply_chat_template(
        messages,
        reasoning_level="none",
        add_generation_prompt=True,
        tokenize=False
    )
    
    assert formatted.endswith("<|think_none|><|think_end|>")

def test_tool_calls():
    """Tool call formatting"""
    tokenizer = NanochatTokenizer("path/to/tokenizer")
    
    messages = [
        Message(
            role="assistant",
            reasoning="Need current time",
            reasoning_level="medium",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="get_time",
                    arguments={}
                )
            ]
        ),
    ]
    
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    
    assert "<|tool_call|>" in formatted
    assert '"id":"call_1"' in formatted or "'id': 'call_1'" in formatted
    assert "<|tool_call_end|>" in formatted

def test_training_inference_consistency():
    """Training and inference use same formatting"""
    tokenizer = NanochatTokenizer("path/to/tokenizer")
    
    messages = [
        Message(role="user", content="Test"),
        Message(
            role="assistant",
            reasoning="Thinking",
            reasoning_level="medium",
            content="Response"
        ),
    ]
    
    # Format for training
    train_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True
    )
    
    # Format for inference (without last message)
    inference_ids = tokenizer.apply_chat_template(
        messages[:-1],
        add_generation_prompt=True,
        reasoning_level="medium",
        tokenize=True
    )
    
    # Inference prompt should be prefix of training sequence
    assert train_ids[:len(inference_ids)] == inference_ids
```

**Success criteria**:
- ✓ All tests pass
- ✓ Training/inference consistency verified
- ✓ Edge cases handled

---

### 5.2 Integration Test

**File**: `tests/test_end_to_end.py` (new)

Test complete pipeline:
1. Create synthetic conversation data
2. Format with chat template
3. Train small model (few steps)
4. Generate with different reasoning levels
5. Verify format consistency

---

## Phase 6: Migration & Documentation

### 6.1 Retrain Tokenizer

```bash
# Train new tokenizer with control tokens
python -m scripts.tok_train --vocab-size 32000 --control-tokens

# Verify
python -m scripts.tok_eval
```

### 6.2 Update All Scripts

**Scripts to update**:
- ✓ `scripts/chat_sft.py` - use chat template
- ✓ `scripts/chat_rl.py` - use chat template
- ✓ `scripts/chat_eval.py` - use chat template
- ✓ `scripts/chat_cli.py` - use chat template
- ✓ Remove all manual formatting code

### 6.3 Documentation

**File**: `docs/CHAT_PROTOCOL.md` (new)

Document:
- Token protocol version
- Message schema
- Chat template usage
- Reasoning levels
- Tool calling format
- Migration guide from old format

---

## Implementation Checklist

### Phase 1: Token Architecture ✓
- [ ] Create `token_protocol.py` with token definitions
- [ ] Update `tok_train.py` to train 32K vocab + control tokens
- [ ] Create `messages.py` with Message schema
- [ ] Test: Tokenizer has 32256 tokens at correct IDs

### Phase 2: Chat Template ✓
- [ ] Create `chat_template.jinja`
- [ ] Implement `apply_chat_template()` in tokenizer
- [ ] Test: Template renders correctly for all message types

### Phase 3: Training Integration ✓
- [ ] Update dataset formatting to use chat template
- [ ] Implement loss masking for reasoning structure
- [ ] Update SFT script
- [ ] Update RL script
- [ ] Update evaluation scripts
- [ ] Test: Training runs with new format

### Phase 4: Inference Integration ✓
- [ ] Update CLI to use chat template
- [ ] Implement response parsing
- [ ] Support reasoning level control
- [ ] Test: Generation works with all reasoning levels

### Phase 5: HF Compatibility ✓
- [ ] Create export script
- [ ] Generate tokenizer artifacts
- [ ] Test: Load with AutoTokenizer
- [ ] Test: apply_chat_template works in HF

### Phase 6: Testing ✓
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Verify training/inference consistency
- [ ] Test all reasoning levels
- [ ] Test tool calling

### Phase 7: Documentation ✓
- [ ] Write protocol documentation
- [ ] Document migration guide
- [ ] Add usage examples
- [ ] Update README

---

## Timeline Estimate

- **Phase 1 (Token Architecture)**: 2-3 hours
- **Phase 2 (Chat Template)**: 2-3 hours
- **Phase 3 (Training Integration)**: 3-4 hours
- **Phase 4 (Inference Integration)**: 2-3 hours
- **Phase 5 (HF Compatibility)**: 1-2 hours
- **Phase 6 (Testing)**: 2-3 hours
- **Phase 7 (Documentation)**: 1-2 hours

**Total**: ~15-20 hours of focused implementation

---

## Success Criteria

At completion, Nanochat will have:

1. ✓ **Clean token protocol**: 32K vocab + 256 control tokens
2. ✓ **Single source of truth**: Chat template for all formatting
3. ✓ **Reasoning support**: Explicit none/low/medium/high levels
4. ✓ **Tool calling**: Structured tool call format
5. ✓ **HF compatible**: Standard tokenizer artifacts
6. ✓ **Future-proof**: Reserved tokens for expansion
7. ✓ **Tested**: Unit + integration tests pass
8. ✓ **Documented**: Clear protocol documentation

And training can proceed with:
- Reasoning datasets at different levels
- Tool-use / agent datasets
- Future: Multimodal datasets (vision tokens already reserved)

---

## Notes

- **Backward compatibility**: Old checkpoints won't work with new tokenizer (expected)
- **Migration**: Will need to retrain base model with new tokenizer
- **Cost**: ~26M tokens for 100-step base training @ 262K batch size
- **Benefit**: Clean foundation for all future Nanochat development

---

## Next Steps After Implementation

1. Retrain base model (100-200 steps) with new tokenizer
2. Create reasoning datasets for each level (none/low/medium/high)
3. SFT training with mixed reasoning levels
4. Create tool-use datasets
5. Agent training with tool calling
6. (Future) Add vision encoder and multimodal training

This gives you a solid foundation for building the reasoning/agent model you described!
