"""
Nanochat Token Protocol Definition
Version: 1.0.0

This module defines the token namespace and protocol for Nanochat models.
The protocol separates lexical tokens (BPE vocabulary) from control tokens
(chat structure, reasoning, tools, multimodal placeholders).

Token Ranges:
    0-31999     : Lexical vocabulary (BPE-trained)
    32000-32255 : Control tokens (reserved namespace)
        32000-32009  : Core tokens (pad, bos, eos, unk)
        32010-32019  : Message structure
        32020-32029  : Role markers
        32030-32039  : Reasoning control
        32040-32049  : Tool calling
        32050-32059  : Multimodal (reserved for future)
        32060-32255  : Reserved for expansion

Design Philosophy:
    - Small vocabulary ≠ limited capability
    - Control tokens are cheap (256 IDs vs 125M params)
    - Future-proof: Reserved space allows non-breaking expansion
    - No hardcoded IDs elsewhere in codebase
"""

from enum import IntEnum
from typing import Literal

# Token ranges
LEXICAL_VOCAB_SIZE = 32000  # Base BPE vocabulary (0-31999)
CONTROL_TOKEN_BASE = 32000
CONTROL_TOKEN_LIMIT = 32256
TOTAL_VOCAB_SIZE = 32256

# Protocol version (semantic versioning)
PROTOCOL_VERSION = "1.0.0"


class ControlTokenID(IntEnum):
    """
    Control token IDs - never hardcode these elsewhere in codebase.
    Always use this enum or SPECIAL_TOKENS dict to reference control tokens.
    """
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
    
    # Multimodal placeholders (reserved for future, 32050-32059)
    IMAGE = 32050
    IMAGE_END = 32051
    VIDEO = 32052
    VIDEO_END = 32053
    AUDIO = 32054
    AUDIO_END = 32055
    
    # Reserved for future expansion: 32060-32255
    # These IDs exist in the vocabulary but have no semantic meaning yet


# Token strings mapping
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
    
    # Multimodal (reserved for future)
    "image": "<|image|>",
    "image_end": "<|image_end|>",
    "video": "<|video|>",
    "video_end": "<|video_end|>",
    "audio": "<|audio|>",
    "audio_end": "<|audio_end|>",
}

# Create reserved token placeholders (32060-32255)
# These exist in vocab but are unused - available for future protocol versions
RESERVED_TOKENS = {
    f"reserved_{i:03d}": f"<|reserved_{i:03d}|>"
    for i in range(60, 256)
}

# Inverse mapping: token string -> ID
TOKEN_TO_ID = {
    token: getattr(ControlTokenID, name.upper())
    for name, token in SPECIAL_TOKENS.items()
    if hasattr(ControlTokenID, name.upper())
}

# Valid enums for message validation
Role = Literal["system", "developer", "user", "assistant", "tool"]
ReasoningLevel = Literal["none", "low", "medium", "high"]

VALID_ROLES = {"system", "developer", "user", "assistant", "tool"}
VALID_REASONING_LEVELS = {"none", "low", "medium", "high"}


def get_all_special_tokens() -> dict[str, str]:
    """
    Get all special tokens including reserved ones.
    Used during tokenizer training to add all control tokens.
    
    Returns:
        Dictionary mapping token names to token strings
    """
    return {**SPECIAL_TOKENS, **RESERVED_TOKENS}


def get_active_special_tokens() -> dict[str, str]:
    """
    Get only active (non-reserved) special tokens.
    Used for tokenizer configuration.
    
    Returns:
        Dictionary mapping token names to token strings
    """
    return SPECIAL_TOKENS.copy()


def validate_role(role: str) -> None:
    """
    Validate that a role is valid.
    
    Args:
        role: Role string to validate
        
    Raises:
        ValueError: If role is invalid
    """
    if role not in VALID_ROLES:
        raise ValueError(
            f"Invalid role '{role}'. Must be one of: {VALID_ROLES}"
        )


def validate_reasoning_level(level: str) -> None:
    """
    Validate that a reasoning level is valid.
    
    Args:
        level: Reasoning level string to validate
        
    Raises:
        ValueError: If reasoning level is invalid
    """
    if level not in VALID_REASONING_LEVELS:
        raise ValueError(
            f"Invalid reasoning_level '{level}'. "
            f"Must be one of: {VALID_REASONING_LEVELS}"
        )


def get_protocol_info() -> dict:
    """
    Get protocol metadata for checkpoints and documentation.
    
    Returns:
        Dictionary with protocol information
    """
    return {
        "protocol_version": PROTOCOL_VERSION,
        "lexical_vocab_size": LEXICAL_VOCAB_SIZE,
        "control_token_base": CONTROL_TOKEN_BASE,
        "control_token_limit": CONTROL_TOKEN_LIMIT,
        "total_vocab_size": TOTAL_VOCAB_SIZE,
        "active_special_tokens": len(SPECIAL_TOKENS),
        "reserved_tokens": len(RESERVED_TOKENS),
    }


if __name__ == "__main__":
    # Print protocol information
    print("Nanochat Token Protocol")
    print("=" * 50)
    info = get_protocol_info()
    for key, value in info.items():
        print(f"{key:25s}: {value}")
    
    print("\nToken ID Ranges:")
    print(f"  Lexical vocab    : 0 - {LEXICAL_VOCAB_SIZE-1}")
    print(f"  Core tokens      : {ControlTokenID.PAD} - {ControlTokenID.UNK}")
    print(f"  Message structure: {ControlTokenID.MESSAGE_START} - {ControlTokenID.MESSAGE_END}")
    print(f"  Roles            : {ControlTokenID.SYSTEM} - {ControlTokenID.TOOL}")
    print(f"  Reasoning        : {ControlTokenID.THINK_NONE} - {ControlTokenID.THINK_END}")
    print(f"  Tool calls       : {ControlTokenID.TOOL_CALL} - {ControlTokenID.TOOL_RESULT_END}")
    print(f"  Multimodal (rsv) : {ControlTokenID.IMAGE} - {ControlTokenID.AUDIO_END}")
    print(f"  Reserved         : 32060 - 32255")
    
    print(f"\nSpecial Tokens: {len(SPECIAL_TOKENS)}")
    for name, token in sorted(SPECIAL_TOKENS.items()):
        token_id = TOKEN_TO_ID.get(token, "N/A")
        print(f"  {token_id:5d} {name:20s} {token}")
    
    print(f"\nReserved Tokens: {len(RESERVED_TOKENS)} (32060-32255)")
