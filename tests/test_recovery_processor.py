"""
Tests for the RecoveryProcessor class.

This module contains comprehensive tests for the RecoveryProcessor class,
including unit tests for reprocessing logic, error handling, progress tracking,
and integration with the ParallelLLMProcessor.
"""

import pytest
import time
import logging
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import List, Tuple, Dict

from src.llmtools.recovery_processor import RecoveryProcessor
from src.llmtools.parallel_llm_processor import ParallelLLMProcessor


class TestRecoveryProcessor:
    """Test suite for RecoveryProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a mock chat function
        self.mock_chat_fn = Mock()
        self.mock_chat_fn.__name__ = "mock_chat_fn"
        
        # Create a mock ParallelLLMProcessor
        self.mock_processor = Mock(spec=ParallelLLMProcessor)
        self.mock_processor.chat_fn = self.mock_chat_fn
        self.mock_processor.num_workers = 2
        self.mock_processor.retry_attempts = 3
        self.mock_processor.retry_delay = 0.1
        self.mock_processor.timeout = 5.0
        
        # Create RecoveryProcessor instance
        self.recovery_processor = RecoveryProcessor(self.mock_processor)
    
    def test_initialization(self):
        """Test RecoveryProcessor initialization."""
        assert self.recovery_processor.processor == self.mock_processor
        assert self.recovery_processor.chat_fn == self.mock_chat_fn
        assert self.recovery_processor.num_workers == 2
        assert self.recovery_processor.retry_attempts == 3
        assert self.recovery_processor.retry_delay == 0.1
        assert self.recovery_processor.timeout == 5.0
    
    def test_process_failed_prompts_empty_list(self):
        """Test processing empty list of failed records."""
        result = self.recovery_processor.process_failed_prompts([])
        assert result == {}
    
    def test_process_failed_prompts_single_success(self):
        """Test successful processing of a single failed record."""
        # Setup mock to return a successful result
        self.mock_chat_fn.return_value = "Successful response"
        
        failed_records = [(1, "Test prompt")]
        result = self.recovery_processor.process_failed_prompts(failed_records)
        
        assert result == {1: "Successful response"}
        self.mock_chat_fn.assert_called_once_with("Test prompt")
    
    def test_process_failed_prompts_multiple_success(self):
        """Test successful processing of multiple failed records."""
        # Setup mock to return different responses based on input
        def mock_response(prompt):
            return f"Response to: {prompt}"
        
        self.mock_chat_fn.side_effect = mock_response
        
        failed_records = [
            (1, "First prompt"),
            (2, "Second prompt"),
            (3, "Third prompt")
        ]
        
        result = self.recovery_processor.process_failed_prompts(failed_records)
        
        expected = {
            1: "Response to: First prompt",
            2: "Response to: Second prompt", 
            3: "Response to: Third prompt"
        }
        assert result == expected
        assert self.mock_chat_fn.call_count == 3
    
    def test_process_failed_prompts_with_failures(self):
        """Test processing with some failures."""
        # Setup mock to fail based on prompt content
        def mock_response(prompt):
            if "Prompt 1" in prompt:
                return "Success 1"
            elif "Prompt 2" in prompt:
                raise Exception("API Error")
            elif "Prompt 3" in prompt:
                return "Success 3"
            else:
                return "Default response"
        
        self.mock_chat_fn.side_effect = mock_response
        
        failed_records = [
            (1, "Prompt 1"),
            (2, "Prompt 2"),
            (3, "Prompt 3")
        ]
        
        result = self.recovery_processor.process_failed_prompts(failed_records)
        
        assert result[1] == "Success 1"
        assert "Reprocessing failed: Failed after 3 attempts: API Error" in result[2]
        assert result[3] == "Success 3"
    
    def test_process_single_failed_record_success(self):
        """Test successful processing of a single record."""
        self.mock_chat_fn.return_value = "Test response"
        
        result = self.recovery_processor._process_single_failed_record(1, "Test prompt")
        
        assert result == "Test response"
        self.mock_chat_fn.assert_called_once_with("Test prompt")
    
    def test_process_single_failed_record_with_retries(self):
        """Test processing with retries before success."""
        # First two calls fail, third succeeds
        self.mock_chat_fn.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            "Success on third try"
        ]
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = self.recovery_processor._process_single_failed_record(1, "Test prompt")
        
        assert result == "Success on third try"
        assert self.mock_chat_fn.call_count == 3
    
    def test_process_single_failed_record_all_retries_fail(self):
        """Test processing when all retry attempts fail."""
        self.mock_chat_fn.side_effect = Exception("Persistent failure")
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            with pytest.raises(Exception) as exc_info:
                self.recovery_processor._process_single_failed_record(1, "Test prompt")
        
        assert "Failed after 3 attempts" in str(exc_info.value)
        assert self.mock_chat_fn.call_count == 3
    
    def test_process_single_failed_record_timeout(self):
        """Test processing with timeout."""
        # Create a processor with very short timeout
        self.recovery_processor.timeout = 0.1
        
        # Mock chat function that takes longer than timeout
        def slow_response(prompt):
            time.sleep(0.2)
            return "Too slow"
        
        self.mock_chat_fn.side_effect = slow_response
        
        with pytest.raises(Exception) as exc_info:
            self.recovery_processor._process_single_failed_record(1, "Test prompt")
        
        assert "Failed after" in str(exc_info.value)
    
    def test_exponential_backoff_delay(self):
        """Test that retry delays follow exponential backoff."""
        self.mock_chat_fn.side_effect = [
            Exception("Fail 1"),
            Exception("Fail 2"),
            "Success"
        ]
        
        with patch('time.sleep') as mock_sleep:
            result = self.recovery_processor._process_single_failed_record(1, "Test prompt")
        
        # Check that sleep was called with exponential backoff delays
        expected_delays = [0.1, 0.2]  # retry_delay * 2^(attempt-1)
        actual_delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert actual_delays == expected_delays
        assert result == "Success"
    
    def test_get_processing_stats(self):
        """Test getting processing configuration statistics."""
        stats = self.recovery_processor.get_processing_stats()
        
        expected_stats = {
            'num_workers': 2,
            'retry_attempts': 3,
            'retry_delay': 0.1,
            'timeout': 5.0,
            'chat_function': 'mock_chat_fn'
        }
        
        assert stats == expected_stats
    
    def test_validate_configuration_valid(self):
        """Test configuration validation with valid settings."""
        assert self.recovery_processor.validate_configuration() is True
    
    def test_validate_configuration_invalid_chat_fn(self):
        """Test configuration validation with invalid chat function."""
        self.recovery_processor.chat_fn = "not_callable"
        assert self.recovery_processor.validate_configuration() is False
    
    def test_validate_configuration_invalid_num_workers(self):
        """Test configuration validation with invalid num_workers."""
        self.recovery_processor.num_workers = 0
        assert self.recovery_processor.validate_configuration() is False
    
    def test_validate_configuration_invalid_retry_attempts(self):
        """Test configuration validation with invalid retry_attempts."""
        self.recovery_processor.retry_attempts = -1
        assert self.recovery_processor.validate_configuration() is False
    
    def test_validate_configuration_invalid_retry_delay(self):
        """Test configuration validation with invalid retry_delay."""
        self.recovery_processor.retry_delay = -1.0
        assert self.recovery_processor.validate_configuration() is False
    
    def test_validate_configuration_invalid_timeout(self):
        """Test configuration validation with invalid timeout."""
        self.recovery_processor.timeout = 0
        assert self.recovery_processor.validate_configuration() is False
    
    @patch('src.llmtools.recovery_processor.tqdm')
    def test_progress_bar_integration(self, mock_tqdm):
        """Test that progress bar is properly integrated."""
        # Setup mock progress bar
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_pbar
        
        self.mock_chat_fn.return_value = "Response"
        
        failed_records = [(1, "Prompt 1"), (2, "Prompt 2")]
        self.recovery_processor.process_failed_prompts(failed_records)
        
        # Verify progress bar was created with enhanced format
        mock_tqdm.assert_called_once_with(
            total=2, 
            desc="Reprocessing 2 failed prompts",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        assert mock_pbar.update.call_count == 2
        # Verify set_description was called to update progress
        assert mock_pbar.set_description.call_count >= 2
    
    def test_logging_integration(self, caplog):
        """Test that appropriate logging messages are generated."""
        with caplog.at_level(logging.INFO):
            self.mock_chat_fn.return_value = "Response"
            
            failed_records = [(1, "Test prompt")]
            self.recovery_processor.process_failed_prompts(failed_records)
        
        # Check that appropriate log messages were generated (updated for enhanced logging)
        log_messages = [record.message for record in caplog.records]
        assert any("Initializing reprocessing of 1 failed records" in msg for msg in log_messages)
        assert any("Reprocessing configuration:" in msg for msg in log_messages)
        assert any("Reprocessing phase completed" in msg for msg in log_messages)
        assert any("Processing statistics:" in msg for msg in log_messages)
        assert any("Successful reprocessing: 1" in msg for msg in log_messages)
        assert any("Failed reprocessing: 0" in msg for msg in log_messages)
    
    def test_concurrent_processing(self):
        """Test that processing uses concurrent execution."""
        # Create a chat function that records call times
        call_times = []
        
        def timed_chat_fn(prompt):
            call_times.append(time.time())
            time.sleep(0.1)  # Small delay to test concurrency
            return f"Response to {prompt}"
        
        self.mock_chat_fn.side_effect = timed_chat_fn
        
        failed_records = [(1, "Prompt 1"), (2, "Prompt 2")]
        
        start_time = time.time()
        result = self.recovery_processor.process_failed_prompts(failed_records)
        end_time = time.time()
        
        # With 2 workers and 2 prompts, should complete faster than sequential
        # Sequential would take ~0.2s, concurrent should be closer to ~0.1s
        assert end_time - start_time < 0.15
        assert len(call_times) == 2
        
        # Verify results
        assert result[1] == "Response to Prompt 1"
        assert result[2] == "Response to Prompt 2"
    
    def test_error_handling_in_process_failed_prompts(self):
        """Test error handling in the main processing method."""
        # Mock ThreadPoolExecutor to raise an exception
        with patch('src.llmtools.recovery_processor.ThreadPoolExecutor') as mock_executor:
            mock_executor.side_effect = Exception("Executor failed")
            
            with pytest.raises(RuntimeError) as exc_info:
                self.recovery_processor.process_failed_prompts([(1, "Test")])
            
            assert "Reprocessing operation failed" in str(exc_info.value)
    
    def test_integration_with_real_parallel_processor(self):
        """Test integration with a real ParallelLLMProcessor instance."""
        def simple_chat_fn(prompt):
            return f"Processed: {prompt}"
        
        # Create real processor
        real_processor = ParallelLLMProcessor(
            chat_fn=simple_chat_fn,
            num_workers=2,
            retry_attempts=2,
            retry_delay=0.05,
            timeout=1.0
        )
        
        # Create recovery processor
        recovery_processor = RecoveryProcessor(real_processor)
        
        # Test processing
        failed_records = [(1, "Hello"), (2, "World")]
        result = recovery_processor.process_failed_prompts(failed_records)
        
        assert result == {1: "Processed: Hello", 2: "Processed: World"}
        
        # Clean up
        real_processor.close()


class TestRecoveryProcessorEdgeCases:
    """Test edge cases and boundary conditions for RecoveryProcessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_chat_fn = Mock()
        self.mock_processor = Mock(spec=ParallelLLMProcessor)
        self.mock_processor.chat_fn = self.mock_chat_fn
        self.mock_processor.num_workers = 1
        self.mock_processor.retry_attempts = 1
        self.mock_processor.retry_delay = 0.01
        self.mock_processor.timeout = 1.0
        
        self.recovery_processor = RecoveryProcessor(self.mock_processor)
    
    def test_large_number_of_failed_records(self):
        """Test processing a large number of failed records."""
        # Create 100 failed records
        failed_records = [(i, f"Prompt {i}") for i in range(1, 101)]
        
        def mock_response(prompt):
            return f"Response to {prompt}"
        
        self.mock_chat_fn.side_effect = mock_response
        
        result = self.recovery_processor.process_failed_prompts(failed_records)
        
        assert len(result) == 100
        assert all(f"Response to Prompt {i}" == result[i] for i in range(1, 101))
    
    def test_mixed_success_and_failure_scenarios(self):
        """Test mixed scenarios with various types of failures."""
        failed_records = [
            (1, "Success prompt"),
            (2, "Timeout prompt"),
            (3, "Exception prompt"),
            (4, "Another success")
        ]
        
        def mock_response(prompt):
            if "Success prompt" in prompt or "Another success" in prompt:
                return f"Success: {prompt}"
            elif "Timeout" in prompt:
                raise FutureTimeoutError("Request timed out")
            elif "Exception" in prompt:
                raise ValueError("Invalid input")
            else:
                return f"Default: {prompt}"
        
        self.mock_chat_fn.side_effect = mock_response
        
        result = self.recovery_processor.process_failed_prompts(failed_records)
        
        assert result[1] == "Success: Success prompt"
        assert "Reprocessing failed: Failed after 1 attempts: Request timed out" in result[2]
        assert "Reprocessing failed: Failed after 1 attempts: Invalid input" in result[3]
        assert result[4] == "Success: Another success"
    
    def test_zero_retry_attempts(self):
        """Test behavior with zero retry attempts."""
        self.recovery_processor.retry_attempts = 0
        self.mock_chat_fn.side_effect = Exception("Immediate failure")
        
        with pytest.raises(Exception) as exc_info:
            self.recovery_processor._process_single_failed_record(1, "Test")
        
        assert "Failed after 0 attempts: No retry attempts configured" in str(exc_info.value)
        assert self.mock_chat_fn.call_count == 0  # No attempts made
    
    def test_very_long_prompts(self):
        """Test processing with very long prompts."""
        long_prompt = "A" * 10000  # 10KB prompt
        self.mock_chat_fn.return_value = "Response to long prompt"
        
        result = self.recovery_processor._process_single_failed_record(1, long_prompt)
        
        assert result == "Response to long prompt"
        self.mock_chat_fn.assert_called_once_with(long_prompt)
    
    def test_unicode_and_special_characters(self):
        """Test processing prompts with unicode and special characters."""
        special_prompts = [
            (1, "Hello 世界 🌍"),
            (2, "Café naïve résumé"),
            (3, "Math: ∑∞ ∫ ∂ ≠ ≤ ≥"),
            (4, "Symbols: @#$%^&*()[]{}|\\:;\"'<>?,./")
        ]
        
        def echo_response(prompt):
            return f"Echo: {prompt}"
        
        self.mock_chat_fn.side_effect = echo_response
        
        result = self.recovery_processor.process_failed_prompts(special_prompts)
        
        for record_id, prompt in special_prompts:
            assert result[record_id] == f"Echo: {prompt}"