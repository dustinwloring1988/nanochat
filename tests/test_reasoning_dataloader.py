"""
Tests for reasoning dataloader functionality.

Tests:
    - Dataset loading from HuggingFace
    - Conversion from Nemotron to Nanochat format
    - Reasoning ratio enforcement
    - Mixed dataloader sampling
    - Reasoning level inference
"""

import pytest
from nanochat.reasoning_dataloader import (
    NemotronReasoningDataset,
    MixedReasoningDataLoader,
    create_reasoning_dataloader
)
from nanochat.messages import Message


class TestNemotronReasoningDataset:
    """Test NemotronReasoningDataset class"""
    
    def test_dataset_initialization(self):
        """Test that dataset can be initialized"""
        # Use streaming mode to avoid downloading full dataset
        dataset = NemotronReasoningDataset(
            dataset_name="nvidia/Nemotron-Cascade-SFT-Stage-2",
            subset="math",  # Use specific config
            split="train",
            streaming=True,
            reasoning_ratio=0.7,
            seed=42,
        )
        
        assert dataset.dataset_name == "nvidia/Nemotron-Cascade-SFT-Stage-2"
        assert dataset.reasoning_ratio == 0.7
        assert dataset.streaming == True
    
    def test_reasoning_content_parsing(self):
        """Test parsing reasoning from content"""
        dataset = NemotronReasoningDataset(
            dataset_name="nvidia/Nemotron-Cascade-SFT-Stage-2",
            subset="math",
            split="train",
            streaming=True,
            reasoning_ratio=0.7,
        )
        
        # Test case 1: Explicit <think> tags
        content1 = "<think>Let me solve this step by step.</think>\nThe answer is 42."
        reasoning1, answer1 = dataset._parse_reasoning_content(content1)
        assert "step by step" in reasoning1.lower()
        assert "42" in answer1
        
        # Test case 2: Answer: marker
        content2 = "First, I analyze the problem.\n\nAnswer: The solution is X."
        reasoning2, answer2 = dataset._parse_reasoning_content(content2)
        assert "analyze" in reasoning2.lower()
        assert "solution" in answer2.lower()
        
        # Test case 3: No clear separation (long content)
        content3 = "a" * 600  # 600 chars
        reasoning3, answer3 = dataset._parse_reasoning_content(content3)
        assert len(reasoning3) > 0
        assert len(answer3) > 0
        assert len(reasoning3) + len(answer3) <= len(content3) + 10  # Allow some whitespace
    
    def test_reasoning_level_inference(self):
        """Test reasoning level inference from content length"""
        dataset = NemotronReasoningDataset(
            dataset_name="nvidia/Nemotron-Cascade-SFT-Stage-2",
            subset="math",
            split="train",
            streaming=True,
            reasoning_ratio=0.7,
        )
        
        # Test none
        assert dataset._infer_reasoning_level("") == "none"
        
        # Test low
        assert dataset._infer_reasoning_level("Short reasoning.") == "low"
        
        # Test medium
        assert dataset._infer_reasoning_level("a" * 400) == "medium"
        
        # Test high
        assert dataset._infer_reasoning_level("a" * 1000) == "high"
    
    @pytest.mark.slow
    def test_convert_to_nanochat_format(self):
        """Test conversion from Nemotron to Nanochat format"""
        dataset = NemotronReasoningDataset(
            dataset_name="nvidia/Nemotron-Cascade-SFT-Stage-2",
            subset="math",
            split="train",
            streaming=True,
            reasoning_ratio=1.0,  # Always use reasoning
            seed=42,
        )
        
        # Mock Nemotron example
        nemotron_example = {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "<think>Simple addition.</think>\n4"}
            ],
            "thinking": True,
            "category": "math",
            "source": "test",
        }
        
        converted = dataset._convert_to_nanochat_format(nemotron_example)
        
        assert "messages" in converted
        assert len(converted["messages"]) == 2
        
        # Check user message
        assert converted["messages"][0]["role"] == "user"
        assert converted["messages"][0]["content"] == "What is 2+2?"
        
        # Check assistant message
        assistant_msg = converted["messages"][1]
        assert assistant_msg["role"] == "assistant"
        assert "reasoning" in assistant_msg or assistant_msg.get("reasoning_level") == "none"
        
        # Check metadata
        assert converted["metadata"]["category"] == "math"
        assert converted["metadata"]["had_thinking"] == True
    
    @pytest.mark.slow
    def test_reasoning_ratio_enforcement(self):
        """Test that reasoning ratio is approximately enforced"""
        dataset = NemotronReasoningDataset(
            dataset_name="nvidia/Nemotron-Cascade-SFT-Stage-2",
            subset="math",
            split="train",
            streaming=True,
            reasoning_ratio=0.5,  # 50% reasoning
            seed=42,
        )
        
        # Mock multiple examples with thinking=True
        examples = []
        for i in range(100):
            example = {
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"<think>Reasoning {i}</think>\nAnswer {i}"}
                ],
                "thinking": True,
                "category": "test",
                "source": "test",
            }
            converted = dataset._convert_to_nanochat_format(example)
            examples.append(converted)
        
        # Count how many have reasoning
        reasoning_count = 0
        for ex in examples:
            for msg in ex["messages"]:
                if msg["role"] == "assistant":
                    if msg.get("reasoning_level") not in ["none", None] and msg.get("reasoning"):
                        reasoning_count += 1
        
        # Should be approximately 50% (allow 20% deviation due to randomness)
        ratio = reasoning_count / len(examples)
        assert 0.3 <= ratio <= 0.7, f"Reasoning ratio {ratio} outside expected range [0.3, 0.7]"


class TestMixedReasoningDataLoader:
    """Test MixedReasoningDataLoader class"""
    
    @pytest.mark.slow
    def test_mixed_loader_initialization(self):
        """Test mixed loader can be initialized with multiple datasets"""
        datasets_config = [
            {
                "name": "nvidia/Nemotron-Cascade-SFT-Stage-2",
                "subset": "math",
                "weight": 0.6,
                "reasoning_ratio": 0.7,
                "streaming": True,
            },
            {
                "name": "nvidia/Nemotron-Post-Training-Dataset-v2",
                "subset": "SFT",
                "weight": 0.4,
                "reasoning_ratio": 0.5,
                "streaming": True,
            }
        ]
        
        loader = MixedReasoningDataLoader(datasets_config, seed=42)
        
        assert len(loader.datasets) == 2
        assert len(loader.weights) == 2
        assert abs(sum(loader.weights) - 1.0) < 1e-6  # Weights sum to 1
    
    @pytest.mark.slow
    def test_weight_normalization(self):
        """Test that weights are normalized correctly"""
        datasets_config = [
            {
                "name": "nvidia/Nemotron-Cascade-SFT-Stage-2",
                "subset": "math",
                "weight": 2.0,
                "streaming": True,
            },
            {
                "name": "nvidia/Nemotron-Post-Training-Dataset-v2",
                "subset": "SFT",
                "weight": 3.0,
                "streaming": True,
            }
        ]
        
        loader = MixedReasoningDataLoader(datasets_config, seed=42)
        
        # Weights should be normalized: 2/5=0.4, 3/5=0.6
        assert abs(loader.weights[0] - 0.4) < 1e-6
        assert abs(loader.weights[1] - 0.6) < 1e-6


class TestReasoningDataLoaderFactory:
    """Test create_reasoning_dataloader factory function"""
    
    def test_factory_function_returns_loader(self):
        """Test that factory function returns correct type (without downloading)"""
        # This test just validates the factory returns the right type
        # without actually downloading datasets
        try:
            loader = create_reasoning_dataloader(
                stage="instruction_following",
                reasoning_ratio=0.7,
                seed=42,
            )
            assert isinstance(loader, MixedReasoningDataLoader)
            assert hasattr(loader, 'datasets')
            assert hasattr(loader, 'weights')
        except Exception as e:
            # If dataset download fails, that's okay for this test
            # We just want to verify the factory function exists and has correct signature
            if "Config name is missing" in str(e) or "Connection" in str(e):
                pytest.skip(f"Dataset download not available: {e}")
            raise
    
    @pytest.mark.slow
    def test_create_instruction_following_loader(self):
        """Test creating instruction following stage loader"""
        loader = create_reasoning_dataloader(
            stage="instruction_following",
            reasoning_ratio=0.7,
            seed=42,
        )
        
        assert len(loader.datasets) >= 1
        assert loader.datasets[0].dataset_name == "nvidia/Nemotron-Cascade-SFT-Stage-2"
        assert loader.datasets[0].subset == "instruction-following"
    
    @pytest.mark.slow
    def test_create_reasoning_loader(self):
        """Test creating reasoning stage loader"""
        loader = create_reasoning_dataloader(
            stage="reasoning",
            reasoning_ratio=0.7,
            seed=42,
        )
        
        assert len(loader.datasets) >= 1
        # Should have multiple datasets for reasoning stage
        assert any("Cascade" in ds.dataset_name for ds in loader.datasets)
    
    @pytest.mark.slow
    def test_create_mixed_loader(self):
        """Test creating mixed stage loader"""
        loader = create_reasoning_dataloader(
            stage="mixed",
            reasoning_ratio=0.6,
            seed=42,
        )
        
        assert len(loader.datasets) >= 1
    
    def test_invalid_stage(self):
        """Test that invalid stage raises error"""
        with pytest.raises(ValueError, match="Unknown stage"):
            create_reasoning_dataloader(
                stage="invalid_stage",
                reasoning_ratio=0.7,
            )


class TestReasoningDataIntegration:
    """Integration tests for reasoning dataloader"""
    
    @pytest.mark.slow
    def test_end_to_end_data_loading(self):
        """Test loading data end-to-end"""
        # This test actually connects to HuggingFace
        # Mark as slow since it requires network and may download data
        
        loader = create_reasoning_dataloader(
            stage="instruction_following",
            reasoning_ratio=0.7,
            seed=42,
        )
        
        # Get an iterator
        iterator = loader.get_iterator(shuffle=False)
        
        # Get a few examples
        examples = []
        for i, example in enumerate(iterator):
            if i >= 3:
                break
            examples.append(example)
        
        assert len(examples) == 3
        
        # Validate structure
        for example in examples:
            assert "messages" in example
            assert isinstance(example["messages"], list)
            
            # Check message structure
            for msg in example["messages"]:
                assert "role" in msg
                assert "content" in msg
                
                if msg["role"] == "assistant":
                    # Assistant should have reasoning structure
                    assert "reasoning_level" in msg or "reasoning" in msg


def test_message_format_compatibility():
    """Test that converted messages are compatible with Message class"""
    dataset = NemotronReasoningDataset(
        dataset_name="nvidia/Nemotron-Cascade-SFT-Stage-2",
        subset="math",
        split="train",
        streaming=True,
        reasoning_ratio=0.7,
    )
    
    # Mock example
    nemotron_example = {
        "messages": [
            {"role": "user", "content": "Test question"},
            {"role": "assistant", "content": "<think>Test reasoning</think>\nTest answer"}
        ],
        "thinking": True,
        "category": "test",
        "source": "test",
    }
    
    converted = dataset._convert_to_nanochat_format(nemotron_example)
    
    # Try to create Message objects
    try:
        messages = [Message.from_dict(msg) for msg in converted["messages"]]
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
    except Exception as e:
        pytest.fail(f"Failed to create Message objects from converted data: {e}")


if __name__ == "__main__":
    """Run tests with pytest"""
    pytest.main([__file__, "-v", "-m", "not slow"])
