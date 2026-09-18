"""
Chat Tokenizer with Template Support

This module extends the base RustBPETokenizer with chat template functionality
according to the Nanochat token protocol v1.0.0.

Key features:
    - apply_chat_template() for structured message formatting
    - Single source of truth for training and inference
    - Support for reasoning levels, tool calls, and multimodal placeholders
    - Loss masking for training
"""

import os
from pathlib import Path
from typing import List, Union, Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader

from nanochat.tokenizer import RustBPETokenizer
from nanochat.token_protocol import (
    SPECIAL_TOKENS, RESERVED_TOKENS, 
    TOTAL_VOCAB_SIZE, CONTROL_TOKEN_BASE,
    get_all_special_tokens, PROTOCOL_VERSION,
    validate_reasoning_level
)
from nanochat.messages import Message, ToolCall


class ChatTokenizer(RustBPETokenizer):
    """
    Enhanced tokenizer with chat template support.
    
    This extends RustBPETokenizer to support:
    - Structured message formatting via Jinja2 templates
    - Reasoning levels (none/low/medium/high)
    - Tool calling
    - Multimodal content placeholders
    - Protocol-compliant token namespace
    
    Usage:
        tokenizer = ChatTokenizer.from_directory("path/to/tokenizer")
        
        # Format messages for training/inference
        messages = [
            Message(role="user", content="What is 2+2?"),
            Message(
                role="assistant",
                reasoning="Simple arithmetic",
                reasoning_level="low",
                content="4"
            )
        ]
        
        token_ids = tokenizer.apply_chat_template(
            messages,
            reasoning_level="low",
            add_generation_prompt=False,
            tokenize=True
        )
    """
    
    def __init__(self, enc, bos_token):
        super().__init__(enc, bos_token)
        
        # Load chat template
        template_path = Path(__file__).parent / "chat_template.jinja"
        if not template_path.exists():
            raise FileNotFoundError(
                f"Chat template not found: {template_path}. "
                "Ensure chat_template.jinja is in the nanochat package."
            )
        
        env = Environment(loader=FileSystemLoader(template_path.parent))
        self.chat_template = env.get_template(template_path.name)
        
        # Cache special token IDs for fast lookup
        self._special_token_ids = {}
        self._build_token_id_cache()
    
    def _build_token_id_cache(self):
        """Build cache of special token name -> ID mappings"""
        for name, token_str in SPECIAL_TOKENS.items():
            try:
                token_id = self.encode_special(token_str)
                self._special_token_ids[name] = token_id
            except Exception as e:
                # Token might not exist in vocab yet (during training)
                pass
    
    def get_special_token_id(self, name: str) -> int:
        """
        Get ID for a special token by name.
        
        Args:
            name: Special token name (e.g., "assistant", "think_medium")
            
        Returns:
            Token ID
            
        Raises:
            ValueError: If token name is unknown
        """
        if name not in self._special_token_ids:
            # Try to encode it
            if name in SPECIAL_TOKENS:
                token_str = SPECIAL_TOKENS[name]
                self._special_token_ids[name] = self.encode_special(token_str)
            else:
                raise ValueError(f"Unknown special token: {name}")
        
        return self._special_token_ids[name]
    
    def apply_chat_template(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        reasoning_level: str = "medium",
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        return_dict: bool = False,
    ) -> Union[List[int], str, Dict[str, Any]]:
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
            
        Example:
            # For training
            tokens = tokenizer.apply_chat_template(
                messages=[
                    Message(role="user", content="Hello"),
                    Message(role="assistant", reasoning="Greeting", 
                           reasoning_level="low", content="Hi!")
                ],
                tokenize=True
            )
            
            # For inference
            tokens = tokenizer.apply_chat_template(
                messages=[Message(role="user", content="Hello")],
                reasoning_level="medium",
                add_generation_prompt=True,
                tokenize=True
            )
        """
        # Convert dicts to Message objects
        if messages and isinstance(messages[0], dict):
            messages = [Message.from_dict(m) for m in messages]
        
        # Validate reasoning level
        validate_reasoning_level(reasoning_level)
        
        # Convert messages to dicts for template
        message_dicts = [m.to_dict() for m in messages]
        
        # Render template
        formatted_text = self.chat_template.render(
            messages=message_dicts,
            reasoning_level=reasoning_level,
            add_generation_prompt=add_generation_prompt,
        )
        
        if not tokenize:
            return formatted_text
        
        # Tokenize (use encode_ordinary to not add BOS automatically)
        token_ids = self.encode(formatted_text, prepend=None, append=None)
        
        if return_dict:
            return {
                "input_ids": token_ids,
                "formatted_text": formatted_text,
                "reasoning_level": reasoning_level,
                "attention_mask": [1] * len(token_ids),  # All tokens are attended to
            }
        
        return token_ids
    
    def create_loss_mask(
        self,
        token_ids: List[int],
        mask_input: bool = True,
        train_reasoning: bool = True,
    ) -> List[bool]:
        """
        Create loss mask for training.
        
        Rules:
        - Mask: system, developer, user, tool messages (input)
        - Train: assistant reasoning + content (output)
        - Special handling for reasoning tokens based on mode
        
        Args:
            token_ids: List of token IDs
            mask_input: Whether to mask input (system/user/tool) messages
            train_reasoning: Whether to train on reasoning content
            
        Returns:
            List of booleans (True = train, False = mask)
            
        Example:
            messages = [
                Message(role="user", content="Hello"),
                Message(role="assistant", reasoning="...", 
                       reasoning_level="medium", content="Hi!")
            ]
            token_ids = tokenizer.apply_chat_template(messages, tokenize=True)
            mask = tokenizer.create_loss_mask(token_ids)
        """
        mask = []
        in_assistant = False
        in_reasoning = False
        in_tool_result = False
        
        # Get token IDs for state tracking
        message_start_id = self.get_special_token_id("message_start")
        message_end_id = self.get_special_token_id("message_end")
        assistant_id = self.get_special_token_id("assistant")
        tool_id = self.get_special_token_id("tool")
        
        think_none_id = self.get_special_token_id("think_none")
        think_end_id = self.get_special_token_id("think_end")
        think_low_id = self.get_special_token_id("think_low")
        think_medium_id = self.get_special_token_id("think_medium")
        think_high_id = self.get_special_token_id("think_high")
        
        for token_id in token_ids:
            # Track message boundaries
            if token_id == message_start_id:
                mask.append(False)
                continue
            
            if token_id == message_end_id:
                in_assistant = False
                in_reasoning = False
                in_tool_result = False
                mask.append(False)
                continue
            
            # Track role tokens
            if token_id == assistant_id:
                in_assistant = True
                mask.append(False)  # Mask role token itself
                continue
            
            if token_id == tool_id:
                in_tool_result = True
                mask.append(False)  # Mask role token
                continue
            
            # Handle assistant messages
            if in_assistant:
                # Reasoning control tokens
                if token_id == think_none_id:
                    # Mask <|think_none|> and expect <|think_end|> next
                    in_reasoning = "none"
                    mask.append(False)
                elif token_id in [think_low_id, think_medium_id, think_high_id]:
                    # Start of reasoning content - train if enabled
                    in_reasoning = True
                    mask.append(train_reasoning)
                elif token_id == think_end_id:
                    # End of reasoning block
                    mask.append(train_reasoning if in_reasoning == True else False)
                    in_reasoning = False
                else:
                    # Regular assistant content
                    if in_reasoning == True:
                        # Inside reasoning block
                        mask.append(train_reasoning)
                    else:
                        # Final answer content (always train)
                        mask.append(True)
            
            # Tool results are masked (they're environment observations)
            elif in_tool_result:
                mask.append(False)
            
            # Everything else (system, developer, user) is masked
            else:
                mask.append(False)
        
        return mask
    
    def format_for_training(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        reasoning_level: str = "medium",
        train_reasoning: bool = True,
    ) -> Dict[str, Any]:
        """
        Format a conversation for training with loss mask.
        
        Args:
            messages: List of Message objects or dicts
            reasoning_level: Default reasoning level for assistant messages
            train_reasoning: Whether to train on reasoning content
            
        Returns:
            Dictionary with:
                - input_ids: Token IDs
                - loss_mask: Boolean mask for loss calculation
                - reasoning_level: Reasoning level used
                
        Example:
            batch = tokenizer.format_for_training(
                messages=[
                    Message(role="user", content="What is 2+2?"),
                    Message(role="assistant", reasoning="Simple math",
                           reasoning_level="low", content="4")
                ]
            )
            # Use batch["input_ids"] and batch["loss_mask"] for training
        """
        # Get token IDs
        result = self.apply_chat_template(
            messages=messages,
            reasoning_level=reasoning_level,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
        )
        
        # Create loss mask
        loss_mask = self.create_loss_mask(
            result["input_ids"],
            train_reasoning=train_reasoning
        )
        
        return {
            "input_ids": result["input_ids"],
            "loss_mask": loss_mask,
            "attention_mask": result["attention_mask"],
            "reasoning_level": result["reasoning_level"],
        }
    
    def format_for_generation(
        self,
        messages: List[Union[Message, Dict[str, Any]]],
        reasoning_level: str = "medium",
    ) -> List[int]:
        """
        Format a conversation for generation (inference).
        
        This adds the generation prompt so the model can continue.
        
        Args:
            messages: List of Message objects or dicts (without final assistant message)
            reasoning_level: Reasoning level to use for generation
            
        Returns:
            Token IDs ready for model.generate()
            
        Example:
            prompt_ids = tokenizer.format_for_generation(
                messages=[Message(role="user", content="Hello")],
                reasoning_level="high"
            )
            # Model will generate: <|think_high|>reasoning...<|think_end|>answer<|message_end|>
        """
        return self.apply_chat_template(
            messages=messages,
            reasoning_level=reasoning_level,
            add_generation_prompt=True,
            tokenize=True,
        )
    
    def get_protocol_version(self) -> str:
        """Get the protocol version this tokenizer implements"""
        return PROTOCOL_VERSION
    
    def get_vocab_info(self) -> Dict[str, Any]:
        """
        Get vocabulary information.
        
        Returns:
            Dictionary with vocab stats
        """
        return {
            "protocol_version": PROTOCOL_VERSION,
            "total_vocab_size": self.get_vocab_size(),
            "expected_vocab_size": TOTAL_VOCAB_SIZE,
            "special_tokens": len(SPECIAL_TOKENS),
            "reserved_tokens": len(RESERVED_TOKENS),
            "control_token_base": CONTROL_TOKEN_BASE,
        }


def get_chat_tokenizer(tokenizer_dir: str = None) -> ChatTokenizer:
    """
    Load chat tokenizer from directory.
    
    Args:
        tokenizer_dir: Path to tokenizer directory (default: $NANOCHAT_BASE_DIR/tokenizer)
        
    Returns:
        ChatTokenizer instance
    """
    if tokenizer_dir is None:
        base_dir = os.environ.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat"))
        tokenizer_dir = os.path.join(base_dir, "tokenizer")
    
    return ChatTokenizer.from_directory(tokenizer_dir)


if __name__ == "__main__":
    # Test chat tokenizer functionality
    print("Chat Tokenizer Tests")
    print("=" * 50)
    
    # Note: This requires a trained tokenizer to exist
    # Run scripts/tok_train.py first
    
    try:
        tokenizer = get_chat_tokenizer()
        print(f"✓ Loaded tokenizer")
        print(f"  Protocol version: {tokenizer.get_protocol_version()}")
        
        vocab_info = tokenizer.get_vocab_info()
        print(f"  Vocab size: {vocab_info['total_vocab_size']}")
        print(f"  Special tokens: {vocab_info['special_tokens']}")
        
        # Test message formatting
        messages = [
            Message(role="user", content="What is 2+2?"),
            Message(
                role="assistant",
                reasoning="This is simple arithmetic. 2+2=4.",
                reasoning_level="low",
                content="The answer is 4."
            )
        ]
        
        # Test tokenization
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
        print(f"\n✓ Formatted text preview:")
        print(f"  {formatted_text[:200]}...")
        
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True)
        print(f"\n✓ Token IDs: {len(token_ids)} tokens")
        
        # Test loss mask
        loss_mask = tokenizer.create_loss_mask(token_ids)
        print(f"✓ Loss mask: {sum(loss_mask)}/{len(loss_mask)} tokens to train")
        
        # Test training format
        batch = tokenizer.format_for_training(messages)
        print(f"✓ Training batch keys: {list(batch.keys())}")
        
        # Test generation format
        gen_ids = tokenizer.format_for_generation(
            [messages[0]],  # Just user message
            reasoning_level="medium"
        )
        print(f"✓ Generation prompt: {len(gen_ids)} tokens")
        
        print("\n✓ All chat tokenizer tests passed!")
        
    except FileNotFoundError as e:
        print(f"⚠ Tokenizer not found: {e}")
        print("  Run: python -m scripts.tok_train")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
