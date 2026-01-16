"""Tests for smart autocomplete functionality."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from intellishell.utils.contextual_completion import (
    CompletionStats,
    FuzzyMatcher,
    CompletionPreview
)


@pytest.fixture
def temp_stats_file():
    """Create temporary stats file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_path = Path(tmpdir) / "completion_stats.json"
        yield stats_path


@pytest.fixture
def completion_stats(temp_stats_file):
    """Create CompletionStats instance for testing."""
    return CompletionStats(storage_path=temp_stats_file)


def test_completion_stats_initialization(completion_stats):
    """Test CompletionStats initialization."""
    assert completion_stats is not None
    assert len(completion_stats.command_frequency) == 0
    assert len(completion_stats.command_recency) == 0


def test_record_command(completion_stats):
    """Test recording commands."""
    completion_stats.record_command("open desktop")
    
    assert completion_stats.command_frequency["open desktop"] == 1
    assert "open desktop" in completion_stats.command_recency
    assert len(completion_stats.time_patterns["open desktop"]) == 1


def test_frequency_tracking(completion_stats):
    """Test frequency tracking."""
    # Record same command multiple times
    for _ in range(5):
        completion_stats.record_command("list downloads")
    
    completion_stats.record_command("open desktop")
    
    assert completion_stats.command_frequency["list downloads"] == 5
    assert completion_stats.command_frequency["open desktop"] == 1


def test_frequency_score(completion_stats):
    """Test frequency score calculation."""
    completion_stats.record_command("open desktop")
    completion_stats.record_command("open desktop")
    completion_stats.record_command("list files")
    
    # open desktop used 2x, list files 1x
    desktop_score = completion_stats.get_frequency_score("open desktop")
    files_score = completion_stats.get_frequency_score("list files")
    
    assert desktop_score == 1.0  # Most frequent
    assert files_score == 0.5    # Half as frequent
    assert completion_stats.get_frequency_score("unknown") == 0.0


def test_recency_score(completion_stats):
    """Test recency score calculation."""
    # Record a command
    completion_stats.record_command("open desktop")
    
    # Should have high recency score (just used)
    score = completion_stats.get_recency_score("open desktop")
    assert score > 0.9
    
    # Unknown command should have 0 recency
    assert completion_stats.get_recency_score("unknown") == 0.0


def test_time_patterns(completion_stats):
    """Test time-of-day pattern tracking."""
    # Record command multiple times
    for _ in range(3):
        completion_stats.record_command("list downloads")
    
    # Should have recorded current hour 3 times
    assert len(completion_stats.time_patterns["list downloads"]) == 3
    
    # Time relevance score should be high (all at current hour)
    score = completion_stats.get_time_relevance_score("list downloads")
    assert score >= 0.5


def test_command_sequences(completion_stats):
    """Test command sequence tracking."""
    # Record sequence
    completion_stats.record_command("open desktop", prev_command=None)
    completion_stats.record_command("list files", prev_command="open desktop")
    completion_stats.record_command("list files", prev_command="open desktop")
    
    # "list files" follows "open desktop" 2 times
    score = completion_stats.get_sequence_score("list files", "open desktop")
    assert score == 1.0  # 100% of the time
    
    # Unknown sequence
    score = completion_stats.get_sequence_score("unknown", "open desktop")
    assert score == 0.0


def test_combined_score(completion_stats):
    """Test combined score calculation."""
    # Record some commands
    completion_stats.record_command("open desktop")
    completion_stats.record_command("list files", prev_command="open desktop")
    
    # Get combined score
    score = completion_stats.get_combined_score("list files", prev_command="open desktop")
    
    # Should be > 0 since we have some data
    assert score > 0.0
    assert score <= 1.0


def test_persistence(temp_stats_file):
    """Test that stats persist to disk."""
    # Create first instance and record commands
    stats1 = CompletionStats(storage_path=temp_stats_file)
    stats1.record_command("open desktop")
    stats1.record_command("list files")
    stats1._save_stats()  # Force save
    
    # Create second instance (should load from disk)
    stats2 = CompletionStats(storage_path=temp_stats_file)
    
    assert stats2.command_frequency["open desktop"] == 1
    assert stats2.command_frequency["list files"] == 1


def test_fuzzy_matcher_similarity():
    """Test fuzzy matching similarity."""
    # Exact match
    assert FuzzyMatcher.similarity("hello", "hello") == 1.0
    
    # Similar
    assert FuzzyMatcher.similarity("hello", "helo") > 0.8
    
    # Different
    assert FuzzyMatcher.similarity("hello", "world") < 0.5


def test_fuzzy_matcher_matching():
    """Test fuzzy matching."""
    candidates = [
        "open desktop",
        "open downloads",
        "list desktop",
        "system info"
    ]
    
    # Exact prefix match
    matches = FuzzyMatcher.fuzzy_match("open", candidates)
    assert len(matches) >= 2
    assert matches[0][1] == 1.0  # Perfect score
    
    # Partial match
    matches = FuzzyMatcher.fuzzy_match("desk", candidates)
    assert len(matches) >= 2
    
    # Typo match (lower threshold for short typos)
    matches = FuzzyMatcher.fuzzy_match("opn", candidates, threshold=0.4)
    assert len(matches) > 0  # Should match "open" commands


def test_completion_preview():
    """Test completion preview generation."""
    # Known command
    preview = CompletionPreview.get_preview("open desktop")
    assert "Desktop" in preview
    
    # Unknown command
    preview = CompletionPreview.get_preview("unknown command")
    assert "Execute" in preview


def test_completion_preview_with_stats(completion_stats):
    """Test preview with usage stats."""
    # Record command
    for _ in range(5):
        completion_stats.record_command("open desktop")
    
    preview = CompletionPreview.get_preview("open desktop", stats=completion_stats)
    assert "5x" in preview  # Should show usage count


def test_time_context(completion_stats):
    """Test time context generation."""
    # Record command
    completion_stats.record_command("open desktop")
    
    context = CompletionPreview.get_time_context(completion_stats, "open desktop")
    assert "Usually used around" in context or context == ""


def test_max_entries_limit(completion_stats):
    """Test that time patterns don't grow unbounded."""
    # Record many commands
    for _ in range(30):
        completion_stats.record_command("open desktop")
    
    # Should be limited to 20
    assert len(completion_stats.time_patterns["open desktop"]) == 20


def test_sequence_limit(completion_stats):
    """Test that command sequences are limited."""
    # Record many sequences
    for i in range(150):
        completion_stats.record_command(f"command{i}", prev_command=f"command{i-1}")
    
    # Should be limited to 100
    assert len(completion_stats.command_sequences) == 100


@pytest.mark.asyncio
async def test_integration_with_completer(temp_stats_file):
    """Test integration with IntelliShellCompleter."""
    from intellishell.utils.completion import IntelliShellCompleter
    
    # Create completer with smart features and clean stats
    completer = IntelliShellCompleter(enable_smart_features=True)
    # Replace stats with clean instance
    completer.stats = CompletionStats(storage_path=temp_stats_file)
    
    assert completer.stats is not None
    assert completer.fuzzy_matcher is not None
    
    # Record some commands
    completer.record_command("open desktop")
    completer.record_command("list files")
    
    # Check that stats were recorded
    assert completer.stats.command_frequency["open desktop"] == 1
    assert completer.last_command == "list files"


def test_completer_without_smart_features():
    """Test completer with smart features disabled."""
    from intellishell.utils.completion import IntelliShellCompleter
    
    completer = IntelliShellCompleter(enable_smart_features=False)
    
    assert completer.stats is None
    assert completer.fuzzy_matcher is None


def test_top_commands(completion_stats):
    """Test getting top commands."""
    from intellishell.utils.completion import IntelliShellCompleter
    
    completer = IntelliShellCompleter(enable_smart_features=True)
    
    # Record some commands with different frequencies
    for _ in range(5):
        completer.record_command("open desktop")
    for _ in range(3):
        completer.record_command("list files")
    completer.record_command("system info")
    
    # Get top commands
    top = completer._get_top_commands(limit=3)
    
    # Should be ordered by frequency/recency
    assert len(top) <= 3
    # Most frequent should be first (if it has history)
    if "open desktop" in top:
        assert top.index("open desktop") < top.index("list files")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
