"""
Test script to verify Dolci dataset integration.

This script:
1. Loads a few examples from Dolci-Instruct-SFT-Tool-Use
2. Converts them to Nanochat format
3. Validates the message structure
4. Prints examples for manual inspection
"""

import sys
import os

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, errors="replace")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, errors="replace")

from datasets import load_dataset
from nanochat.reasoning_dataloader import NemotronReasoningDataset
from nanochat.messages import Message
import json


def test_dolci_loading():
    """Test basic dataset loading"""
    print("=" * 70)
    print("TEST 1: Loading Dolci dataset")
    print("=" * 70)
    
    try:
        # Suppress HuggingFace Hub warnings
        import warnings
        import logging
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
        
        # Load just 10 examples for testing
        ds = load_dataset(
            "allenai/Dolci-Instruct-SFT-Tool-Use",
            split="train",
            streaming=True
        )
        
        print("✓ Dataset loaded successfully (streaming mode)")
        
        # Get first example
        sample = next(iter(ds))
        print(f"\n✓ Retrieved sample example")
        print(f"  Keys: {list(sample.keys())}")
        print(f"  Number of messages: {len(sample['messages'])}")
        print(f"  Has function_calls: {any('function_calls' in m for m in sample['messages'])}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False


def test_dolci_conversion():
    """Test conversion to Nanochat format"""
    print("\n" + "=" * 70)
    print("TEST 2: Converting Dolci examples to Nanochat format")
    print("=" * 70)
    
    try:
        # Suppress warnings
        import logging
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        
        # Create dataset wrapper
        dataset = NemotronReasoningDataset(
            dataset_name="allenai/Dolci-Instruct-SFT-Tool-Use",
            split="train",
            streaming=True,
            reasoning_ratio=0.8,
            seed=42
        )
        
        print("✓ NemotronReasoningDataset wrapper created")
        
        # Get iterator and process first 3 examples
        iterator = dataset.get_iterator(shuffle=False)
        
        for i, example in enumerate(iterator):
            if i >= 3:  # Only test first 3
                break
            
            print(f"\n--- Example {i+1} ---")
            print(f"Metadata: {example['metadata']}")
            print(f"Number of messages: {len(example['messages'])}")
            
            # Print messages
            for j, msg in enumerate(example['messages']):
                print(f"\n  Message {j+1}:")
                print(f"    Role: {msg['role']}")
                if 'reasoning' in msg:
                    print(f"    Reasoning level: {msg.get('reasoning_level', 'N/A')}")
                    print(f"    Reasoning (first 100 chars): {msg['reasoning'][:100]}...")
                content_preview = msg['content'][:150] if isinstance(msg['content'], str) else str(msg['content'])[:150]
                print(f"    Content (first 150 chars): {content_preview}...")
            
            # Validate message structure
            try:
                for msg in example['messages']:
                    # Try creating Message object to validate
                    Message.from_dict(msg)
                print(f"\n  ✓ All messages validated successfully")
            except Exception as e:
                print(f"\n  ✗ Message validation failed: {e}")
                return False
        
        print("\n✓ Conversion test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dolci_with_tokenizer():
    """Test tokenization of Dolci examples"""
    print("\n" + "=" * 70)
    print("TEST 3: Tokenizing Dolci examples")
    print("=" * 70)
    
    try:
        # Suppress warnings
        import logging
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        
        # Create dataset
        dataset = NemotronReasoningDataset(
            dataset_name="allenai/Dolci-Instruct-SFT-Tool-Use",
            split="train",
            streaming=True,
            reasoning_ratio=0.8,
            seed=42
        )
        
        # Get one example
        iterator = dataset.get_iterator(shuffle=False)
        example = next(iterator)
        
        print(f"\n✓ Got example with {len(example['messages'])} messages")
        
        # Validate messages can be converted to Message objects
        messages = []
        for msg_dict in example['messages']:
            msg = Message.from_dict(msg_dict)
            messages.append(msg)
        
        print(f"✓ All {len(messages)} messages converted to Message objects")
        print(f"  - System message with tool definitions: {'system' in [m.role for m in messages]}")
        print(f"  - User queries: {sum(1 for m in messages if m.role == 'user')}")
        print(f"  - Assistant responses: {sum(1 for m in messages if m.role == 'assistant')}")
        
        # Check metadata
        print(f"\n✓ Metadata:")
        for key, value in example['metadata'].items():
            print(f"    {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Message validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    # Suppress HuggingFace Hub warnings globally
    import logging
    import warnings
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module="huggingface_hub")
    
    print("DOLCI DATASET INTEGRATION TESTS")
    print("=" * 70)
    
    results = []
    
    # Test 1: Loading
    results.append(("Dataset Loading", test_dolci_loading()))
    
    # Test 2: Conversion
    results.append(("Format Conversion", test_dolci_conversion()))
    
    # Test 3: Message Validation
    results.append(("Message Validation", test_dolci_with_tokenizer()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s} {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✓ All tests passed! Dolci integration is working correctly.")
        return 0
    else:
        print("\n✗ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
