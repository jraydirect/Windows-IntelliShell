# IntelliShell v0.1

**Windows-first semantic command shell with AI reasoning, self-healing execution, and vector memory.**

Natural language interface for Windows with LLM-powered intent routing, automatic error recovery, and semantic command history.

## Key Features

- **Natural Language Commands** - "open brave", "list 20 processes", "what did I do yesterday?"
- **AI-Powered Routing** - Ollama integration for understanding conversational queries
- **Self-Healing Execution** - Automatic error detection and repair with human-in-the-loop
- **Semantic Memory** - Vector-based command history search (optional ChromaDB)
- **Clipboard History** - Persistent clipboard manager with search and restore
- **Smart Autocomplete** - Context-aware tab completion with frequency tracking and previews
- **Tab Completion** - Intelligent autocomplete that learns from your usage patterns
- **Process Management** - List, monitor, and manage Windows processes
- **File Operations** - Navigate folders, list files, watch for changes
- **Application Launcher** - Open any Windows application by name

## Quick Start

```bash
# Install
pip install -e .

# Optional: Full features (semantic memory, notifications, etc.)
pip install -e ".[full]"

# Run
ishell
```

## Usage Examples

### File Operations

```bash
intellishell> open desktop
intellishell> list 5 most recent items in downloads
intellishell> watch downloads for pdf
```

### Process Management

```bash
intellishell> list 20 processes
intellishell> kill notepad
intellishell> check admin
```

### Application Launching

```bash
intellishell> open brave
intellishell> open discord
intellishell> open cursor
intellishell> open notepad
```

### Natural Language Queries

```bash
intellishell> what are my recent commands
intellishell> show me 15 processes currently running
intellishell> list 7 files in my downloads folder
```

### System Information

```bash
intellishell> system info
intellishell> get hostname
intellishell> disk space
```

## Architecture

```
User Input
    |
    v
[Natural Language Detection]
    |
    +---> [LLM Router (Ollama)] ---> Intent Match
    |
    +---> [Rule-Based Parser] ------> Intent Match
    |
    v
[Execution Planner]
    |
    v
[Self-Healing Executor]
    |
    +---> Try Execute
    |
    +---> Detect Error
    |
    +---> AI Repair Suggestion
    |
    +---> Human Approval
    |
    +---> Retry
    |
    v
[Provider Execution]
    |
    +---> FileSystem Provider
    +---> App Provider
    +---> System Provider
    +---> Watch Provider
    +---> Doctor Provider
    +---> Memory Provider
    +---> Clipboard Provider
```

## AI Bridge (Ollama)

IntelliShell uses Ollama for natural language understanding:

```bash
# Natural language queries are automatically routed to LLM
intellishell> show me my computer name
# LLM interprets -> get_hostname intent

intellishell> what processes are using the most memory
# LLM interprets -> list_processes intent
```

**Setup:**
1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3:8b`
3. IntelliShell auto-detects and uses it

**Disable AI:** `ishell --no-ai`

## Self-Healing Execution

Automatic error recovery with AI-powered repair suggestions:

```bash
intellishell> opn desktp
# Detects typo, suggests correction
# "Did you mean: open desktop?"
```

When commands fail, the self-healing executor:
1. Detects the error type
2. Generates repair suggestion via LLM
3. Asks for human approval
4. Retries with corrected command

**Disable self-healing:** `ishell --no-self-healing`

## Semantic Memory (Optional)

Vector-based command history for semantic search:

```bash
# Install ChromaDB
pip install chromadb

# Then use semantic queries
intellishell> what folder did I open yesterday?
intellishell> what did I do with downloads?
intellishell> recent memories
```

Commands are automatically indexed as vector embeddings in `~/.intellishell/vector_store`.

**Disable memory:** `ishell --no-memory`

## Clipboard History Manager

Persistent clipboard history with automatic monitoring and search:

```bash
# View clipboard history (last 20 entries)
intellishell> clipboard history

# Search clipboard history
intellishell> clipboard search "that API key"
intellishell> clipboard search password

# Restore previous clipboard entry
intellishell> clipboard restore 5

# View statistics
intellishell> clipboard stats

# Control monitoring
intellishell> clipboard start monitoring
intellishell> clipboard stop monitoring

# Clear history
intellishell> clipboard clear
```

**Features:**
- Automatic background monitoring of clipboard changes
- Persistent storage to `~/.intellishell/clipboard_history.jsonl`
- Search through clipboard history
- Restore any previous clipboard entry
- Deduplication (skips consecutive duplicates)
- Size limits (default: 100 entries, 10MB max)
- Thread-safe operations

**Storage Location:** `~/.intellishell/clipboard_history.jsonl`

## Smart Autocomplete with Context

Intelligent tab completion that learns from your usage patterns and provides contextual suggestions:

**Features:**
- **Frequency-Based Ranking** - Most-used commands appear first
- **Recency Scoring** - Recently used commands get priority
- **Time-of-Day Awareness** - Suggests commands you typically use at this hour
- **Command Sequence Learning** - Predicts next command based on previous one
- **Fuzzy Matching** - Handles typos and partial matches
- **Preview Generation** - Shows what each command does
- **Usage Statistics** - Displays how often you've used each command

**How It Works:**
```bash
# Press TAB for completions
intellishell> op[TAB]
# Shows: open desktop (used 15x) | Usually used around 9am
#        open downloads (used 8x)
#        open documents (used 3x)

# Fuzzy matching handles typos
intellishell> lst[TAB]
# Shows: list downloads
#        list desktop
#        list files

# Context-aware suggestions
intellishell> open desktop
intellishell> [TAB]
# Shows: list files (often follows "open desktop")
#        list desktop
```

**Learning:**
- Automatically tracks command frequency
- Records time-of-day patterns
- Learns command sequences
- Persists to `~/.intellishell/completion_stats.json`

**Privacy:**
- All data stored locally
- No telemetry or cloud sync
- Can be cleared anytime

## Available Commands

### FileSystem Provider
- `open desktop` - Open Desktop folder
- `open downloads` - Open Downloads folder
- `open documents` - Open Documents folder
- `open recycle bin` - Open Recycle Bin
- `open explorer` - Open File Explorer
- `list files` - List files in current directory
- `list downloads` - List files in Downloads
- `list desktop` - List files on Desktop

### App Provider
- `open notepad` - Launch Notepad
- `open calculator` - Launch Calculator
- `open settings` - Launch Windows Settings
- `open task manager` - Launch Task Manager
- `open control panel` - Launch Control Panel
- `open [app name]` - Launch any application (brave, discord, chrome, cursor, etc.)

### System Provider
- `list processes` - List top 10 processes by memory
- `list [N] processes` - List N processes
- `kill process [PID]` - Kill process by PID
- `kill [name]` - Kill process by name
- `most memory` - Show top memory consumer
- `check admin` - Check if running as Administrator

### SystemMonitor Provider
- `system info` - Display system information
- `get hostname` - Show computer name
- `get username` - Show current user
- `disk space` - Show disk usage

### Watch Provider
- `watch downloads` - Monitor Downloads folder
- `watch downloads for pdf` - Monitor for specific file type
- `list watches` - Show active file monitors
- `stop watching` - Stop all file monitors

### Doctor Provider
- `check system health` - Run system diagnostics
- `check dependencies` - Verify installed dependencies

### Memory Provider (requires ChromaDB)
- `what did I [action]` - Semantic search of command history
- `recent memories` - Show recent command memories

### Clipboard Provider
- `clipboard history` - Show clipboard history (last 20 entries)
- `clipboard search <query>` - Search clipboard history
- `clipboard restore N` - Restore entry N to clipboard
- `clipboard clear` - Clear clipboard history
- `clipboard stats` - Show clipboard statistics
- `clipboard start monitoring` - Start background clipboard monitoring
- `clipboard stop monitoring` - Stop clipboard monitoring

### Special Commands
- `help` - Show available commands
- `history` - Show command history
- `!N` - Replay command N from history
- `stats` - Show session statistics
- `manifest` - Show detailed system manifest
- `clear` / `cls` - Clear screen
- `exit` / `quit` - Exit shell

## Command-Line Options

```bash
# Interactive mode (default)
ishell

# Single command execution
ishell -c "open desktop"

# Debug mode (show intent matching details)
ishell --debug

# Disable AI bridge
ishell --no-ai

# Disable semantic memory
ishell --no-memory

# Custom log file
ishell --log-file /path/to/log.txt

# Show version
ishell --version
```

## Configuration Files

```
~/.intellishell/
├── vector_store/          # ChromaDB vector storage (if enabled)
├── logs/
│   └── shell.log         # Structured logs
├── clipboard_history.jsonl  # Clipboard history storage
├── completion_stats.json    # Smart autocomplete learning data
├── history.jsonl         # Transaction log
└── repairs.jsonl         # Self-healing repair log
```

## Transaction Logging

Every command is logged to `~/.intellishell/history.jsonl`:

```json
{
  "timestamp": "2026-01-16T10:30:45.123456",
  "user_input": "open desktop",
  "intent_name": "open_desktop",
  "provider_name": "filesystem",
  "confidence": 0.95,
  "success": true,
  "entities": [],
  "metadata": {}
}
```

## Dependencies

### Core (Required)
- `pydantic>=2.0.0` - Data validation
- `requests>=2.28.0` - HTTP client
- `prompt-toolkit>=3.0.0` - Interactive shell with tab completion

### Optional (Full Install)
- `psutil>=5.9.0` - Process management, disk space
- `pyperclip>=1.8.0` - Clipboard integration
- `rich>=13.0.0` - Enhanced terminal UI
- `watchdog>=3.0.0` - Filesystem monitoring
- `plyer>=2.1.0` or `win10toast>=0.9` - Native notifications
- `chromadb>=0.4.0` - Semantic memory (vector storage)

**Install all features:**
```bash
pip install -e ".[full]"
```

## Safety Features

- **Read-Only Default** - Most operations are non-destructive
- **Critical Process Protection** - Cannot kill system processes
- **Admin Detection** - Warns when elevated privileges needed
- **Human-in-the-Loop** - Destructive actions require confirmation
- **Safety Levels** - GREEN (safe), YELLOW (caution), RED (requires approval)
- **Circuit Breaker** - Prevents repeated failures

## Performance

- **Parser**: <50ms intent matching
- **Entity Extraction**: <10ms
- **LLM Routing**: ~100-500ms (when needed)
- **Command Execution**: Async, non-blocking

## Creating Custom Providers

```python
from intellishell.providers.base import BaseProvider, IntentTrigger, ExecutionResult

class MyProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "my_provider"
    
    @property
    def description(self) -> str:
        return "My custom provider"
    
    def _initialize_triggers(self) -> None:
        self.triggers = [
            IntentTrigger(
                pattern="my command",
                intent_name="do_thing",
                weight=1.0,
                aliases=["alternative command"]
            )
        ]
    
    async def execute(self, intent_name, context=None):
        if intent_name == "do_thing":
            return ExecutionResult(
                success=True,
                message="Thing done!",
                data={"result": "success"}
            )
```

Register in `intellishell/providers/registry.py`:
```python
from intellishell.providers.my_provider import MyProvider

def auto_discover(self, semantic_memory=None):
    providers = [
        # ... existing providers ...
        MyProvider(),
    ]
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Specific test files
pytest tests/test_parser.py -v
pytest tests/test_providers.py -v
pytest tests/test_session.py -v
```

## Troubleshooting

### Semantic Memory Not Available
```bash
pip install chromadb
```

### Ollama Not Found
1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3:8b`
3. Verify: `ollama list`

### Tab Completion Not Working
- Requires `prompt-toolkit>=3.0.0`
- Install: `pip install prompt-toolkit`

### Admin Privileges Required
Some commands (like killing processes) require Administrator privileges:
- Right-click terminal
- Select "Run as Administrator"
- Run `ishell` again

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your provider/feature
4. Add tests
5. Submit a pull request

## Future Roadmap

- Plugin system for external providers
- Cross-platform support (Linux, macOS)
- Voice command integration
- Multi-language support
- Remote providers via RPC
- Advanced process monitoring and analytics
- Integration with Windows Terminal
- Custom theme support
