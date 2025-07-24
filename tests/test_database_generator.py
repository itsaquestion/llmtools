"""
Test Database Generation Utilities

This module provides utilities for creating test databases with various scenarios
for comprehensive testing of the recovery functionality.
"""

import sqlite3
import tempfile
import os
import random
import string
from datetime import datetime, timedelta
from typing import List, Tuple, Any, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class ResultType(Enum):
    """Enumeration of different result types for test data generation."""
    COMPLETE = "complete"
    NULL = "null"
    EMPTY = "empty"
    NA = "na"
    ERROR = "error"


@dataclass
class DatabaseTestRecord:
    """Represents a test record for database generation."""
    id: int
    prompt: str
    result: Optional[str]
    result_type: ResultType
    created_at: Optional[datetime] = None


class TestDatabaseGenerator:
    """Utility class for generating test databases with various scenarios."""
    
    @staticmethod
    def create_temporary_database() -> str:
        """
        Create a temporary database file and return its path.
        
        Returns:
            str: Path to the temporary database file
        """
        temp_fd, temp_path = tempfile.mkstemp(suffix='.db')
        os.close(temp_fd)  # Close file descriptor, we'll use the path
        return temp_path
    
    @staticmethod
    def create_database_with_schema(db_path: str) -> None:
        """
        Create a database with the standard llm_results table schema.
        
        Args:
            db_path: Path to the database file
        """
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE llm_results (
                    id INTEGER PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    result TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    @staticmethod
    def generate_test_prompt(index: int, length: int = 50) -> str:
        """
        Generate a test prompt with specified characteristics.
        
        Args:
            index: Index number for the prompt
            length: Approximate length of the prompt
            
        Returns:
            str: Generated test prompt
        """
        base_prompts = [
            "What is the capital of",
            "Explain the concept of",
            "How do you calculate",
            "What are the benefits of",
            "Describe the process of",
            "What is the difference between",
            "How does",
            "Why is",
            "When should you use",
            "What are the main features of"
        ]
        
        subjects = [
            "machine learning", "quantum computing", "renewable energy",
            "artificial intelligence", "blockchain technology", "data science",
            "cloud computing", "cybersecurity", "biotechnology", "robotics",
            "virtual reality", "augmented reality", "internet of things",
            "big data analytics", "neural networks", "computer vision"
        ]
        
        base = random.choice(base_prompts)
        subject = random.choice(subjects)
        prompt = f"{base} {subject}?"
        
        # Add index for uniqueness
        prompt = f"[{index}] {prompt}"
        
        # Pad or truncate to approximate length
        if len(prompt) < length:
            padding = ''.join(random.choices(string.ascii_letters + ' ', k=length - len(prompt)))
            prompt += f" Additional context: {padding}"
        elif len(prompt) > length:
            prompt = prompt[:length-3] + "..."
        
        return prompt
    
    @staticmethod
    def generate_test_result(result_type: ResultType, prompt: str = "") -> Optional[str]:
        """
        Generate a test result based on the specified type.
        
        Args:
            result_type: Type of result to generate
            prompt: Original prompt (used for generating realistic responses)
            
        Returns:
            Optional[str]: Generated result or None for NULL results
        """
        if result_type == ResultType.NULL:
            return None
        elif result_type == ResultType.EMPTY:
            return ""
        elif result_type == ResultType.NA:
            return "NA"
        elif result_type == ResultType.ERROR:
            error_messages = [
                "Error: API timeout",
                "Error: Rate limit exceeded",
                "Error: Invalid request format",
                "Error: Service unavailable",
                "Error: Authentication failed"
            ]
            return random.choice(error_messages)
        elif result_type == ResultType.COMPLETE:
            # Generate a realistic response based on the prompt
            if "capital" in prompt.lower():
                cities = ["London", "Paris", "Tokyo", "New York", "Berlin", "Sydney"]
                return f"The capital is {random.choice(cities)}."
            elif "calculate" in prompt.lower():
                return "You can calculate this using the standard formula: result = input * factor."
            elif "benefits" in prompt.lower():
                return "The main benefits include improved efficiency, cost reduction, and better outcomes."
            else:
                return f"This is a comprehensive answer to the question about {prompt.split()[-1] if prompt else 'the topic'}."
        else:
            raise ValueError(f"Unknown result type: {result_type}")
    
    @staticmethod
    def create_test_records(
        count: int,
        complete_ratio: float = 0.7,
        null_ratio: float = 0.1,
        empty_ratio: float = 0.1,
        na_ratio: float = 0.05,
        error_ratio: float = 0.05
    ) -> List[DatabaseTestRecord]:
        """
        Create a list of test records with specified distribution of result types.
        
        Args:
            count: Total number of records to create
            complete_ratio: Ratio of complete results
            null_ratio: Ratio of NULL results
            empty_ratio: Ratio of empty string results
            na_ratio: Ratio of "NA" results
            error_ratio: Ratio of error results
            
        Returns:
            List[DatabaseTestRecord]: Generated test records
        """
        # Validate ratios sum to 1.0 (with small tolerance for floating point)
        total_ratio = complete_ratio + null_ratio + empty_ratio + na_ratio + error_ratio
        if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Calculate counts for each type
        complete_count = int(count * complete_ratio)
        null_count = int(count * null_ratio)
        empty_count = int(count * empty_ratio)
        na_count = int(count * na_ratio)
        error_count = int(count * error_ratio)
        
        # Adjust for rounding errors
        remaining = count - (complete_count + null_count + empty_count + na_count + error_count)
        complete_count += remaining
        
        # Create result type list
        result_types = (
            [ResultType.COMPLETE] * complete_count +
            [ResultType.NULL] * null_count +
            [ResultType.EMPTY] * empty_count +
            [ResultType.NA] * na_count +
            [ResultType.ERROR] * error_count
        )
        
        # Shuffle to randomize order
        random.shuffle(result_types)
        
        # Generate records
        records = []
        base_time = datetime.now() - timedelta(hours=1)
        
        for i in range(count):
            prompt = TestDatabaseGenerator.generate_test_prompt(i + 1)
            result_type = result_types[i]
            result = TestDatabaseGenerator.generate_test_result(result_type, prompt)
            created_at = base_time + timedelta(seconds=i * 10)  # 10 seconds apart
            
            records.append(DatabaseTestRecord(
                id=i + 1,
                prompt=prompt,
                result=result,
                result_type=result_type,
                created_at=created_at
            ))
        
        return records
    
    @staticmethod
    def insert_records(db_path: str, records: List[DatabaseTestRecord]) -> None:
        """
        Insert test records into the database.
        
        Args:
            db_path: Path to the database file
            records: List of test records to insert
        """
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            for record in records:
                cursor.execute("""
                    INSERT INTO llm_results (id, prompt, result, created_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    record.id,
                    record.prompt,
                    record.result,
                    record.created_at.isoformat() if record.created_at else None
                ))
            
            conn.commit()
    
    @staticmethod
    def create_test_database_with_scenario(
        scenario_name: str,
        record_count: int = 100,
        **kwargs
    ) -> Tuple[str, List[DatabaseTestRecord]]:
        """
        Create a test database with a specific scenario.
        
        Args:
            scenario_name: Name of the scenario to create
            record_count: Number of records to create
            **kwargs: Additional parameters for record generation
            
        Returns:
            Tuple[str, List[DatabaseTestRecord]]: Database path and list of created records
        """
        db_path = TestDatabaseGenerator.create_temporary_database()
        TestDatabaseGenerator.create_database_with_schema(db_path)
        
        if scenario_name == "mixed_failures":
            # Mixed scenario with various failure types
            records = TestDatabaseGenerator.create_test_records(
                record_count,
                complete_ratio=0.6,
                null_ratio=0.15,
                empty_ratio=0.15,
                na_ratio=0.05,
                error_ratio=0.05,
                **kwargs
            )
        elif scenario_name == "mostly_complete":
            # Mostly complete with few failures
            records = TestDatabaseGenerator.create_test_records(
                record_count,
                complete_ratio=0.95,
                null_ratio=0.02,
                empty_ratio=0.02,
                na_ratio=0.005,
                error_ratio=0.005,
                **kwargs
            )
        elif scenario_name == "mostly_failed":
            # Mostly failed records
            records = TestDatabaseGenerator.create_test_records(
                record_count,
                complete_ratio=0.1,
                null_ratio=0.3,
                empty_ratio=0.3,
                na_ratio=0.15,
                error_ratio=0.15,
                **kwargs
            )
        elif scenario_name == "all_complete":
            # All records complete
            records = TestDatabaseGenerator.create_test_records(
                record_count,
                complete_ratio=1.0,
                null_ratio=0.0,
                empty_ratio=0.0,
                na_ratio=0.0,
                error_ratio=0.0,
                **kwargs
            )
        elif scenario_name == "all_failed":
            # All records failed
            records = TestDatabaseGenerator.create_test_records(
                record_count,
                complete_ratio=0.0,
                null_ratio=0.4,
                empty_ratio=0.3,
                na_ratio=0.2,
                error_ratio=0.1,
                **kwargs
            )
        elif scenario_name == "empty_database":
            # Empty database
            records = []
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        TestDatabaseGenerator.insert_records(db_path, records)
        return db_path, records
    
    @staticmethod
    def get_database_statistics(db_path: str) -> Dict[str, int]:
        """
        Get statistics about the database contents.
        
        Args:
            db_path: Path to the database file
            
        Returns:
            Dict[str, int]: Statistics about the database
        """
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Total records
            cursor.execute("SELECT COUNT(*) FROM llm_results")
            total_records = cursor.fetchone()[0]
            
            # NULL results
            cursor.execute("SELECT COUNT(*) FROM llm_results WHERE result IS NULL")
            null_records = cursor.fetchone()[0]
            
            # Empty string results
            cursor.execute("SELECT COUNT(*) FROM llm_results WHERE result = ''")
            empty_records = cursor.fetchone()[0]
            
            # "NA" results
            cursor.execute("SELECT COUNT(*) FROM llm_results WHERE result = 'NA'")
            na_records = cursor.fetchone()[0]
            
            # Error results
            cursor.execute("SELECT COUNT(*) FROM llm_results WHERE result LIKE 'Error:%'")
            error_records = cursor.fetchone()[0]
            
            # Complete results (including error results, as they are actual responses)
            incomplete_records = null_records + empty_records + na_records
            complete_records = total_records - incomplete_records
            
            return {
                'total_records': total_records,
                'complete_records': complete_records,
                'null_records': null_records,
                'empty_records': empty_records,
                'na_records': na_records,
                'error_records': error_records,
                'incomplete_records': incomplete_records
            }
    
    @staticmethod
    def cleanup_database(db_path: str) -> None:
        """
        Clean up a test database file.
        
        Args:
            db_path: Path to the database file to remove
        """
        if os.path.exists(db_path):
            os.unlink(db_path)


# Convenience functions for common test scenarios

def create_mixed_failure_database(record_count: int = 100) -> Tuple[str, List[DatabaseTestRecord]]:
    """Create a database with mixed failure scenarios."""
    return TestDatabaseGenerator.create_test_database_with_scenario(
        "mixed_failures", record_count
    )


def create_mostly_complete_database(record_count: int = 100) -> Tuple[str, List[DatabaseTestRecord]]:
    """Create a database with mostly complete records."""
    return TestDatabaseGenerator.create_test_database_with_scenario(
        "mostly_complete", record_count
    )


def create_mostly_failed_database(record_count: int = 100) -> Tuple[str, List[DatabaseTestRecord]]:
    """Create a database with mostly failed records."""
    return TestDatabaseGenerator.create_test_database_with_scenario(
        "mostly_failed", record_count
    )


def create_all_complete_database(record_count: int = 100) -> Tuple[str, List[DatabaseTestRecord]]:
    """Create a database with all complete records."""
    return TestDatabaseGenerator.create_test_database_with_scenario(
        "all_complete", record_count
    )


def create_all_failed_database(record_count: int = 100) -> Tuple[str, List[DatabaseTestRecord]]:
    """Create a database with all failed records."""
    return TestDatabaseGenerator.create_test_database_with_scenario(
        "all_failed", record_count
    )


def create_empty_database() -> Tuple[str, List[DatabaseTestRecord]]:
    """Create an empty database."""
    return TestDatabaseGenerator.create_test_database_with_scenario(
        "empty_database", 0
    )


# Basic tests for the database generator functionality

def test_create_mixed_failure_database():
    """Test creating a mixed failure database."""
    db_path, records = create_mixed_failure_database(10)
    
    try:
        # Verify database was created
        assert os.path.exists(db_path)
        
        # Verify records were created
        assert len(records) == 10
        
        # Verify database statistics
        stats = TestDatabaseGenerator.get_database_statistics(db_path)
        assert stats['total_records'] == 10
        assert stats['incomplete_records'] > 0  # Should have some incomplete records
        
    finally:
        TestDatabaseGenerator.cleanup_database(db_path)


def test_create_all_complete_database():
    """Test creating a database with all complete records."""
    db_path, records = create_all_complete_database(5)
    
    try:
        # Verify database was created
        assert os.path.exists(db_path)
        
        # Verify records were created
        assert len(records) == 5
        
        # Verify all records are complete
        stats = TestDatabaseGenerator.get_database_statistics(db_path)
        assert stats['total_records'] == 5
        assert stats['incomplete_records'] == 0
        
    finally:
        TestDatabaseGenerator.cleanup_database(db_path)


def test_create_empty_database():
    """Test creating an empty database."""
    db_path, records = create_empty_database()
    
    try:
        # Verify database was created
        assert os.path.exists(db_path)
        
        # Verify no records
        assert len(records) == 0
        
        # Verify database statistics
        stats = TestDatabaseGenerator.get_database_statistics(db_path)
        assert stats['total_records'] == 0
        
    finally:
        TestDatabaseGenerator.cleanup_database(db_path)


def test_database_statistics():
    """Test database statistics functionality."""
    db_path, records = create_mixed_failure_database(20)
    
    try:
        stats = TestDatabaseGenerator.get_database_statistics(db_path)
        
        # Verify all expected keys are present
        expected_keys = [
            'total_records', 'complete_records', 'null_records',
            'empty_records', 'na_records', 'error_records', 'incomplete_records'
        ]
        
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], int)
            assert stats[key] >= 0
        
        # Verify totals add up correctly
        assert stats['total_records'] == 20
        incomplete_sum = stats['null_records'] + stats['empty_records'] + stats['na_records']
        assert stats['incomplete_records'] == incomplete_sum
        
    finally:
        TestDatabaseGenerator.cleanup_database(db_path)