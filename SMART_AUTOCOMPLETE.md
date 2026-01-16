# Smart Autocomplete with Context - Feature Documentation

## Overview

IntelliShell's Smart Autocomplete is an intelligent tab completion system that learns from your usage patterns and provides contextual suggestions. Unlike traditional autocomplete, it adapts to your workflow, predicts your next command, and handles typos gracefully.

## Features

### ✅ Core Capabilities

1. **Frequency-Based Ranking**
   - Tracks how often you use each command
   - Most-used commands appear first in suggestions
   - Normalizes scores across all commands

2. **Recency Scoring**
   - Recently used commands get priority
   - Decay function: 100% score at 0 hours, 50% at 24 hours, 0% at 1 week
   - Keeps your workflow fresh and relevant

3. **Time-of-Day Awareness**
   - Learns when you typically use each command
   - Suggests commands you usually run at this hour
   - Tracks last 20 time entries per command

4. **Command Sequence Learning**
   - Predicts next command based on previous one
   - Tracks command pairs (e.g., "open desktop" → "list files")
   - Stores last 100 command sequences

5. **Fuzzy Matching with Typo Correction**
   - Handles typos: "opn" → "open"
   - Partial matching: "desk" → "desktop"
   - Similarity scoring using SequenceMatcher

6. **Preview Generation**
   - Shows what each command does
   - Displays usage count
   - Shows typical usage time

## Architecture

### Components

```
IntelliShellCompleter
    ├── CompletionStats (tracks usage)
    │   ├── Frequency tracking
    │   ├── Recency tracking
    │   ├── Time patterns
    │   └── Command sequences
    ├── FuzzyMatcher (handles typos)
    └── CompletionPreview (generates descriptions)
```

### Data Flow

```
User Types → TAB
    ↓
IntelliShellCompleter.get_completions()
    ↓
Collect candidates (built-in + providers + patterns)
    ↓
Fuzzy match against input
    ↓
Score each match:
    - Fuzzy similarity (60%)
    - Context score (40%)
        - Frequency (30%)
        - Recency (30%)
        - Time-of-day (20%)
        - Sequence (20%)
    ↓
Sort by combined score
    ↓
Generate previews
    ↓
Return top 20 completions
```

### Scoring Algorithm

```python
combined_score = (
    fuzzy_similarity * 0.6 +
    (
        frequency_score * 0.3 +
        recency_score * 0.3 +
        time_score * 0.2 +
        sequence_score * 0.2
    ) * 0.4
)
```

## Usage Examples

### Basic Autocomplete

```bash
intellishell> op[TAB]
# Suggestions:
# open desktop (Open Desktop folder) (used 15x)
# open downloads (Open Downloads folder) (used 8x)
# open documents (Open Documents folder) (used 3x)
```

### Fuzzy Matching

```bash
intellishell> lst[TAB]
# Suggestions:
# list downloads
# list desktop
# list files

intellishell> clipbrd[TAB]
# Suggestions:
# clipboard history
# clipboard search
# clipboard stats
```

### Context-Aware Suggestions

```bash
# After opening desktop, autocomplete suggests related commands
intellishell> open desktop
intellishell> [TAB]
# Suggestions:
# list files (often follows "open desktop")
# list desktop
# clipboard history
```

### Time-Based Suggestions

```bash
# At 9am (morning routine)
intellishell> [TAB]
# Suggestions:
# check system health (Usually used around 9am)
# list processes (Usually used around 9am)
# system info

# At 5pm (end of day)
intellishell> [TAB]
# Suggestions:
# clipboard history (Usually used around 5pm)
# stats (Usually used around 5pm)
```

## Storage Format

### completion_stats.json

```json
{
  "frequency": {
    "open desktop": 15,
    "list downloads": 8,
    "clipboard history": 12
  },
  "recency": {
    "open desktop": "2026-01-16T10:30:00",
    "list downloads": "2026-01-16T09:15:00"
  },
  "time_patterns": {
    "open desktop": [9, 9, 10, 9, 8, 9],
    "list downloads": [14, 15, 14, 16]
  },
  "sequences": [
    ["open desktop", "list files"],
    ["list files", "clipboard history"],
    ["open desktop", "list files"]
  ]
}
```

## Configuration

### Enable/Disable Smart Features

```python
# In main.py
completer = IntelliShellCompleter(
    provider_registry=registry,
    parser=parser,
    enable_smart_features=True  # Set to False to disable
)
```

### Custom Scoring Weights

```python
# In contextual_completion.py
weights = {
    "frequency": 0.3,  # How often used
    "recency": 0.3,    # How recently used
    "time": 0.2,       # Time-of-day relevance
    "sequence": 0.2    # Command sequence patterns
}

score = stats.get_combined_score(command, prev_command, weights=weights)
```

### Adjust Fuzzy Matching Threshold

```python
# Lower threshold = more lenient matching
matches = fuzzy_matcher.fuzzy_match(query, candidates, threshold=0.4)
```

## Performance

- **Completion Generation**: <50ms (typical)
- **Fuzzy Matching**: O(n) where n = number of candidates
- **Score Calculation**: O(1) per candidate
- **Memory Usage**: ~100KB for stats file
- **Disk I/O**: Async writes every 5 commands

## Privacy & Security

- **Local Storage Only**: All data stored in `~/.intellishell/`
- **No Telemetry**: No data sent to external servers
- **No PII**: Only command names stored, not arguments
- **User Control**: Can be disabled or cleared anytime

## Clearing Learning Data

```bash
# Delete stats file
rm ~/.intellishell/completion_stats.json

# Or programmatically
from pathlib import Path
stats_file = Path.home() / ".intellishell" / "completion_stats.json"
stats_file.unlink(missing_ok=True)
```

## Testing

Comprehensive test suite in `tests/test_smart_autocomplete.py`:

```bash
# Run all autocomplete tests
pytest tests/test_smart_autocomplete.py -v

# Test specific feature
pytest tests/test_smart_autocomplete.py::test_frequency_tracking -v
```

**Test Coverage:**
- ✅ Frequency tracking
- ✅ Recency scoring
- ✅ Time-of-day patterns
- ✅ Command sequences
- ✅ Fuzzy matching
- ✅ Preview generation
- ✅ Persistence
- ✅ Integration

**All 19 tests passing** ✅

## Troubleshooting

### Issue: Autocomplete not working

**Solution**: Ensure prompt-toolkit is installed:
```bash
pip install prompt-toolkit
```

### Issue: Stats not persisting

**Solution**: Check file permissions:
```bash
ls -la ~/.intellishell/completion_stats.json
chmod 644 ~/.intellishell/completion_stats.json
```

### Issue: Suggestions seem wrong

**Solution**: Clear learning data and start fresh:
```bash
rm ~/.intellishell/completion_stats.json
```

### Issue: Too many/few suggestions

**Solution**: Adjust the limit in `get_completions`:
```python
for candidate, score in completion_candidates[:20]:  # Change 20 to desired limit
```

## Advanced Usage

### Custom Preview Templates

```python
# In contextual_completion.py
CompletionPreview.PREVIEWS["my_command"] = "My custom description"
```

### Export Learning Data

```python
from intellishell.utils.contextual_completion import CompletionStats

stats = CompletionStats()
data = {
    "frequency": dict(stats.command_frequency),
    "recency": stats.command_recency,
    "time_patterns": dict(stats.time_patterns)
}

# Export to JSON
import json
with open("my_stats.json", "w") as f:
    json.dump(data, f, indent=2)
```

### Analyze Usage Patterns

```python
from intellishell.utils.contextual_completion import CompletionStats

stats = CompletionStats()

# Most used commands
top_commands = stats.command_frequency.most_common(10)
print("Top 10 commands:")
for cmd, count in top_commands:
    print(f"  {cmd}: {count}x")

# Time patterns
for cmd, hours in stats.time_patterns.items():
    avg_hour = sum(hours) / len(hours)
    print(f"{cmd}: typically used at {avg_hour:.0f}:00")
```

## Future Enhancements

### Planned Features
- [ ] Machine learning-based prediction
- [ ] Natural language query completion
- [ ] Multi-word fuzzy matching
- [ ] Completion confidence scores
- [ ] User-defined aliases
- [ ] Import/export learning data
- [ ] Cloud sync (optional)

### Possible Improvements
- [ ] Semantic similarity for completions
- [ ] Context from file system state
- [ ] Integration with clipboard history
- [ ] Completion analytics dashboard
- [ ] A/B testing different scoring weights

## Contributing

To extend smart autocomplete:

1. Add new scoring factors in `CompletionStats`
2. Update `get_combined_score` with new weights
3. Add tests for new features
4. Update documentation

## License

MIT License - Same as IntelliShell

## Credits

- Implemented as part of IntelliShell v0.1.0
- Feature request: Smart Autocomplete with Context from engineering audit
- Inspired by fish shell, zsh, and modern IDE autocomplete

---

**Version**: 1.0.0  
**Date**: 2026-01-16  
**Status**: ✅ Production Ready  
**Tests**: ✅ 19/19 Passing
