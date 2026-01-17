"""Async REPL loop with self-healing and production safety."""

import asyncio
import sys
import uuid
from datetime import datetime
from typing import Optional
from intellishell.providers.registry import ProviderRegistry
from intellishell.parser import SemanticParser, IntentMatch, AmbiguousMatch
from intellishell.planner import ExecutionPlanner
from intellishell.session import SessionState
from intellishell.safety import SafetyController
from intellishell.executor import SelfHealingExecutor
from intellishell.utils.clipboard import copy_to_clipboard, should_pipe_to_clipboard, GlobalContext
from intellishell.utils.transaction_log import TransactionLogger
from intellishell.memory import SemanticMemory
from intellishell.ai_bridge import AIBridge
from intellishell.validation import SelfCorrection
from intellishell.utils.terminal import enable_royal_blue_terminal, reset_terminal_color, TerminalColors
from intellishell.utils.display import format_message
import logging

logger = logging.getLogger(__name__)


class IntentShell:
    """
    Async interactive REPL with self-healing and production safety.
    
    Features:
    - Try-repair-retry execution loop
    - Human-in-the-loop safety controls
    - Circuit breaker for failing commands
    - System health diagnostics
    """
    
    def __init__(
        self,
        debug: bool = False,
        enable_ai: bool = True,
        enable_memory: bool = True,
        enable_self_healing: bool = True
    ):
        self.debug = debug
        self.enable_ai = enable_ai
        self.enable_memory = enable_memory
        self.enable_self_healing = enable_self_healing
        self.running = True
        self.prompt = "intellishell> "
        
        # Initialize semantic memory
        self.semantic_memory = None
        if enable_memory:
            try:
                self.semantic_memory = SemanticMemory()
                if self.semantic_memory.is_available():
                    logger.info("Semantic memory initialized")
                else:
                    logger.warning("Semantic memory not available")
                    self.semantic_memory = None
            except Exception as e:
                logger.warning(f"Failed to initialize semantic memory: {e}")
                self.semantic_memory = None
        
        # Initialize clipboard history
        self.clipboard_history = None
        try:
            from intellishell.utils.clipboard import ClipboardHistory
            self.clipboard_history = ClipboardHistory(auto_monitor=True)
            logger.info("Clipboard history initialized with monitoring")
        except Exception as e:
            logger.warning(f"Failed to initialize clipboard history: {e}")
        
        # Initialize provider registry first (needed for AI bridge)
        self.registry = ProviderRegistry()
        self.registry.auto_discover(
            semantic_memory=self.semantic_memory,
            clipboard_history=self.clipboard_history
        )
        
        # Initialize AI bridge with registry for dynamic tool mapping
        self.ai_bridge = None
        if enable_ai:
            try:
                self.ai_bridge = AIBridge(provider_registry=self.registry)
                if self.ai_bridge.is_available():
                    logger.info("AI bridge initialized (Ollama available)")
                else:
                    logger.info("AI bridge initialized but Ollama not running")
            except Exception as e:
                logger.warning(f"Failed to initialize AI bridge: {e}")
        
        # Initialize parser
        self.parser = SemanticParser(self.registry, ai_bridge=self.ai_bridge)
        
        # Initialize safety controller
        self.safety_controller = SafetyController()
        
        # Initialize planner with safety
        self.planner = ExecutionPlanner(self.registry, safety_controller=self.safety_controller)
        
        # Initialize self-healing executor
        self.self_healing_executor = None
        if enable_self_healing and self.ai_bridge:
            self.self_healing_executor = SelfHealingExecutor(
                planner=self.planner,
                ai_bridge=self.ai_bridge
            )
        
        # Initialize session state
        self.session = SessionState(
            session_id=str(uuid.uuid4())[:8],
            start_time=datetime.now()
        )
        
        # Initialize global context
        self.global_context = GlobalContext()
        
        # Initialize transaction logger
        self.transaction_logger = TransactionLogger()
        
        # Initialize self-correction
        self.self_correction = SelfCorrection()
        
        logger.info(f"IntelliShell initialized (session: {self.session.session_id})")
    
    def print_banner(self) -> None:
        """Display welcome banner with gradient yellow dollar signs."""
        from intellishell.utils.terminal import TerminalColors
        
        # ASCII art - exact as provided by user
        banner_lines = [
            r"$$$$$$\            $$\               $$\ $$\ $$\  $$$$$$\  $$\                 $$\ $$\ ",
            r"\_$$  _|           $$ |              $$ |$$ |\__|$$  __$$\ $$ |                $$ |$$ |",
            r"  $$ |  $$$$$$$\ $$$$$$\    $$$$$$\  $$ |$$ |$$\ $$ /  \__|$$$$$$$\   $$$$$$\  $$ |$$ |",
            r"  $$ |  $$  __$$\\_$$  _|  $$  __$$\ $$ |$$ |$$ |\$$$$$$\  $$  __$$\ $$  __$$\ $$ |$$ |",
            r"  $$ |  $$ |  $$ | $$ |    $$$$$$$$ |$$ |$$ |$$ | \____$$\ $$ |  $$ |$$$$$$$$ |$$ |$$ |",
            r"  $$ |  $$ |  $$ | $$ |$$\ $$   ____|$$ |$$ |$$ |$$\   $$ |$$ |  $$ |$$   ____|$$ |$$ |",
            r"$$$$$$\ $$ |  $$ | \$$$$  |\$$$$$$$\ $$ |$$ |$$ |\$$$$$$  |$$ |  $$ |\$$$$$$$\ $$ |$$ |",
            r"\______|\__|  \__|  \____/  \_______|\__|\__|\__| \______/ \__|  \__| \_______|\__|\__|",
        ]
        
        # Gradient yellow colors for dollar signs
        yellow_bright = "\033[38;2;255;255;0m"      # Bright yellow
        yellow_medium = "\033[38;2;255;215;0m"      # Gold/yellow-orange
        yellow_dark = "\033[38;2;255;165;0m"        # Orange-yellow
        reset = TerminalColors.RESET
        
        # Color dollar signs with gradient
        def color_dollar_signs(line: str) -> str:
            """Apply gradient yellow color to dollar signs."""
            if not TerminalColors.supports_color():
                return line
            
            result = ""
            dollar_count = 0
            for char in line:
                if char == '$':
                    # Cycle through gradient colors
                    if dollar_count % 3 == 0:
                        result += yellow_bright + char + reset
                    elif dollar_count % 3 == 1:
                        result += yellow_medium + char + reset
                    else:
                        result += yellow_dark + char + reset
                    dollar_count += 1
                else:
                    result += char
            return result
        
        # Build banner with colored dollar signs
        banner = "\n"
        for line in banner_lines:
            banner += color_dollar_signs(line) + "\n"
        banner += "\n\n"
        banner += "v0.1 - Self-Healing & Production Safety\n"
        banner += "Try-Repair-Retry Loop | HITL Safety | Circuit Breaker\n"
        banner += "Type 'help' for commands, 'check system health' to test\n"
        
        print(banner)
        
        status_parts = []
        if self.semantic_memory and self.semantic_memory.is_available():
            status_parts.append("✓ Semantic Memory")
        else:
            status_parts.append("✗ Semantic Memory")
        
        if self.ai_bridge and self.ai_bridge.is_available():
            status_parts.append("✓ AI Bridge")
        else:
            status_parts.append("✗ AI Bridge")
        
        if self.enable_self_healing:
            status_parts.append("✓ Self-Healing")
        else:
            status_parts.append("✗ Self-Healing")
        
        print(f"Status: {' | '.join(status_parts)}")
        
        if self.debug:
            print(f"\n[DEBUG MODE] Session: {self.session.session_id}")
            print(f"[DEBUG MODE] Providers: {len(self.registry.get_all_providers())}")
        print()
    
    async def process_command(self, user_input: str) -> bool:
        """Process a single user command."""
        if not user_input.strip():
            return True
        
        # Update global clipboard context
        self.global_context.update()
        
        # Check for exit
        if user_input.lower().strip() in ["exit", "quit", "bye"]:
            print("Exiting IntelliShell...")
            return False
        
        # Update global clipboard context (skip if clipboard monitoring is active to avoid duplicate reads)
        if not self.clipboard_history or not self.clipboard_history._monitoring:
            self.global_context.update()
        
        # Special commands (check before parsing)
        user_input_lower = user_input.lower().strip()
        if user_input_lower in ["help", "?"]:
            self._show_help()
            return True
        
        if user_input_lower in ["clear", "cls"]:
            self._clear_screen()
            return True
        
        # Intercept help-related natural language queries
        help_patterns = [
            "what are the available commands",
            "what commands are available",
            "show me the commands",
            "list commands",
            "list the commands",
            "what can I do",
            "show help",
            "show available commands",
            "list commands for me",
            "show commands",
            "available commands",
            "what commands"
        ]
        if any(pattern in user_input_lower for pattern in help_patterns):
            info_msg = format_message("ℹ To see available commands, type 'help' or use the 'manifest' command.", success=False, is_warning=True)
            print(info_msg)
            self._show_help()
            return True
        
        # Intercept history-related natural language queries
        history_patterns = [
            "what are my recent commands",
            "show my recent commands",
            "show my history",
            "what did i do",
            "what did i run",
            "show command history",
            "list my commands",
            "recent commands",
            "my history",
            "command history"
        ]
        if any(pattern in user_input_lower for pattern in history_patterns):
            self._show_history()
            return True
        
        if user_input.lower().strip() in ["stats", "session stats"]:
            self._show_stats()
            return True
        
        if user_input.lower().strip() == "manifest":
            self._show_manifest()
            return True
        
        if user_input.lower().strip() in ["history", "hist"]:
            self._show_history()
            return True
        
        if user_input.startswith("!"):
            return await self._replay_history(user_input)
        
        # Check for clipboard piping
        pipe_to_clipboard, cleaned_input = should_pipe_to_clipboard(user_input)
        
        # Show "thinking" status for natural language queries
        is_nl_query = hasattr(self.parser, '_is_natural_language_query') and \
                     self.parser._is_natural_language_query(cleaned_input)
        if is_nl_query and self.ai_bridge and self.ai_bridge.is_available():
            thinking_msg = format_message(f"[Thinking: Mapping '{cleaned_input[:50]}...' to intent...]", success=False, is_warning=False)
            print(thinking_msg)
        
        # Parse intent
        parse_result = self.parser.parse(
            cleaned_input,
            use_llm_fallback=(self.ai_bridge is not None)
        )
        
        # Handle no match
        if parse_result is None:
            error_msg = format_message(f"Unknown command: '{user_input}'", success=False, is_error=True)
            warning_msg = format_message("Type 'help' for available commands.", success=False, is_warning=True)
            print(error_msg)
            print(warning_msg)
            
            if self.debug:
                print("\n[DEBUG] Top scoring intents:")
                scores = self.parser.get_debug_scores(cleaned_input)
                for intent, pattern, score in scores:
                    print(f"  - {intent}: {pattern} ({score:.2f})")
            
            self.transaction_logger.log_transaction(
                user_input, None, None, 0.0, False
            )
            self.session.add_command(user_input, None, False, 0.0)
            return True
        
        # Handle ambiguous match
        if isinstance(parse_result, AmbiguousMatch):
            self._handle_ambiguity(parse_result)
            return True
        
        # Handle clear match
        intent_match: IntentMatch = parse_result
        
        # Self-correction
        needs_correction, corrected_input, correction_msgs = self.self_correction.validate_and_correct(
            user_input,
            intent_match.intent_name,
            [{"type": e.type, "value": e.value} for e in intent_match.entities]
        )
        
        if needs_correction and correction_msgs:
            for msg in correction_msgs:
                info_msg = format_message(f"ℹ {msg}", success=False, is_warning=True)
                print(info_msg)
        
        # Show debug info
        if self.debug:
            print(f"[DEBUG] Intent: {intent_match.intent_name}")
            print(f"[DEBUG] Provider: {intent_match.provider_name}")
            print(f"[DEBUG] Confidence: {intent_match.confidence:.2f}")
            print(f"[DEBUG] Source: {intent_match.source}")
            print(f"[DEBUG] Safety: {self.safety_controller.get_safety_level(intent_match.intent_name).name}")
            print()
        
        # Execute intent with self-healing
        try:
            # Build context
            execution_context = {
                "original_input": user_input,
                "entities": intent_match.entities,
                "clipboard": self.global_context.get_current(),
                "session_id": self.session.session_id
            }
            
            # Use self-healing executor if available
            if self.self_healing_executor:
                result, repair_attempt = await self.self_healing_executor.execute_with_healing(
                    intent_match,
                    execution_context
                )
                
                if repair_attempt:
                    if self.debug:
                        print(f"[DEBUG] Repair: {repair_attempt.status.value}")
            else:
                result = await self.planner.execute_intent(intent_match, execution_context)
            
            # Display result with color coding
            if result.success:
                # Success messages - already colored in providers, but ensure green for simple success
                print(result.message)
            else:
                # Error messages - color red
                colored_message = format_message(result.message, success=False, is_error=True)
                print(colored_message)
            
            # Pipe to clipboard if requested
            if pipe_to_clipboard and result.success:
                if copy_to_clipboard(result.message):
                    success_msg = format_message("✓ Copied to clipboard", success=True)
                    print(success_msg)
            
            # Store in semantic memory
            if self.semantic_memory and result.success:
                self.semantic_memory.remember(
                    user_input=user_input,
                    intent_name=intent_match.intent_name,
                    provider_name=intent_match.provider_name,
                    output_summary=result.message[:500],
                    confidence=intent_match.confidence,
                    success=True,
                    entities=[{"type": e.type, "value": e.value} for e in intent_match.entities],
                    metadata=result.metadata
                )
            
            # Add to AI context
            if self.ai_bridge:
                self.ai_bridge.add_context(
                    user_input=user_input,
                    intent_name=intent_match.intent_name,
                    result=result.message[:200],
                    entities=[{"type": e.type, "value": e.value} for e in intent_match.entities]
                )
            
            # Log transaction
            self.transaction_logger.log_transaction(
                user_input=user_input,
                intent_name=intent_match.intent_name,
                provider_name=intent_match.provider_name,
                confidence=intent_match.confidence,
                success=result.success,
                result_message=result.message,
                entities=[{"type": e.type, "value": e.value} for e in intent_match.entities],
                metadata=result.metadata
            )
            
            # Track in session
            self.session.add_command(
                user_input,
                intent_match.intent_name,
                result.success,
                intent_match.confidence
            )
            
            # Record for smart autocomplete (if available)
            if hasattr(self, '_completer') and self._completer and result.success:
                try:
                    self._completer.record_command(user_input)
                except Exception as e:
                    logger.debug(f"Failed to record command for autocomplete: {e}")
            
            # Update context
            if result.data and "path" in result.data:
                from pathlib import Path
                self.session.last_directory = Path(result.data["path"])
                if self.ai_bridge:
                    self.ai_bridge.context_manager.update_directory(str(result.data["path"]))
            
        except Exception as e:
            error_msg = format_message(f"Error executing command: {e}", success=False, is_error=True)
            print(error_msg)
            logger.exception("Command execution failed")
            
            self.transaction_logger.log_transaction(
                user_input, intent_match.intent_name, intent_match.provider_name,
                intent_match.confidence, False, str(e)
            )
            self.session.add_command(user_input, intent_match.intent_name, False, intent_match.confidence)
        
        return True
    
    def _handle_ambiguity(self, ambiguous: AmbiguousMatch) -> None:
        """Handle ambiguous matches."""
        print(f"Did you mean...?\n")
        
        for i, suggestion in enumerate(ambiguous.suggestions, 1):
            print(f"  {i}. {suggestion.trigger_pattern} ({suggestion.confidence:.2f})")
            print(f"     → {suggestion.intent_name} via {suggestion.provider_name}")
        
        print(f"\nPlease rephrase your command to be more specific.")
    
    def _clear_screen(self) -> None:
        """Clear the terminal screen and redisplay banner."""
        import os
        # Windows
        if os.name == 'nt':
            os.system('cls')
        # Unix/Linux/Mac
        else:
            os.system('clear')
        
        # Redisplay banner
        self.print_banner()
    
    async def _replay_history(self, command: str) -> bool:
        """Replay a command from history."""
        try:
            index = int(command[1:])
            history = self.transaction_logger.read_history(limit=100)
            
            if index < 1 or index > len(history):
                print(f"Invalid history index: {index}. Valid range: 1-{len(history)}")
                return True
            
            replay_cmd = history[index - 1]["user_input"]
            print(f"Replaying: {replay_cmd}")
            
            return await self.process_command(replay_cmd)
        except ValueError:
            print("Invalid history replay format. Use: !N (e.g., !5)")
            return True
    
    def _show_help(self) -> None:
        """Display help information."""
        print("\nIntelliShell - Available Commands:")
        print("=" * 60)
        
        providers = self.registry.get_all_providers()
        for provider in providers:
            print(f"\n{provider.name.upper()}: {provider.description}")
            triggers = provider.get_triggers()
            for trigger in triggers[:3]:
                print(f"  • {trigger.pattern}")
        
        print("\nSPECIAL COMMANDS:")
        print("  • help / ? - Show this help")
        print("  • clear / cls - Clear screen")
        print("  • stats - Show session statistics")
        print("  • manifest - Show system manifest")
        print("  • history - Show command history")
        print("  • !N - Replay command N")
        print("  • exit / quit - Exit")
        print()
    
    def _show_stats(self) -> None:
        """Display session statistics."""
        stats = self.session.get_stats()
        print("\nSession Statistics:")
        print("=" * 60)
        print(f"Session ID: {stats['session_id']}")
        print(f"Duration: {stats['duration']:.1f} seconds")
        print(f"Total Commands: {stats['total_commands']}")
        print(f"Successful: {stats['successful_commands']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        
        # Safety summary
        safety_summary = self.planner.get_safety_summary()
        print(f"\nSafety:")
        print(f"  Last Action Failed: {safety_summary['last_action_failed']}")
        print(f"  RED Actions: {safety_summary['red_actions']}")
        
        # Circuit breaker status
        if self.self_healing_executor:
            print(f"\nCircuit Breaker: Active")
        
        print()
    
    def _show_manifest(self) -> None:
        """Display system manifest."""
        manifest = self.registry.generate_manifest()
        
        print("\nIntelliShell Manifest:")
        print("=" * 60)
        print(f"Version: {manifest['version']}")
        print(f"Total Commands: {manifest['total_commands']}")
        print(f"Providers: {len(manifest['providers'])}")
        print()
    
    def _show_history(self) -> None:
        """Display command history."""
        history = self.transaction_logger.read_history(limit=20)
        
        if not history:
            print("No command history available.")
            return
        
        print("\nCommand History (last 20):")
        print("=" * 60)
        
        for i, tx in enumerate(history, 1):
            status = "✓" if tx.get("success") else "✗"
            timestamp = tx.get("timestamp", "")[:19]
            cmd = tx.get("user_input", "")
            
            print(f"{i:3d}. {status} [{timestamp}] {cmd}")
        
        print("\nUse !N to replay a command")
        print()
    
    def cleanup(self) -> None:
        """Clean up all resources on shell exit."""
        logger.info("Cleaning up IntelliShell resources...")
        
        # Stop clipboard monitoring
        if self.clipboard_history:
            try:
                self.clipboard_history.stop_monitoring()
                logger.debug("Clipboard monitoring stopped")
            except Exception as e:
                logger.warning(f"Failed to stop clipboard monitoring: {e}")
        
        # Stop file watchers
        try:
            watch_provider = self.registry.get_provider("watch")
            if watch_provider and hasattr(watch_provider, '_active_watches'):
                if watch_provider._active_watches:
                    # Stop all active watches
                    for watch_id, observer in list(watch_provider._active_watches.items()):
                        try:
                            observer.stop()
                            observer.join(timeout=2)
                            if observer.is_alive():
                                logger.warning(f"Watch observer {watch_id} didn't stop within timeout")
                        except Exception as e:
                            logger.warning(f"Failed to stop watch {watch_id}: {e}")
                    watch_provider._active_watches.clear()
                    logger.debug("File watchers stopped")
        except Exception as e:
            logger.warning(f"Failed to stop file watchers: {e}")
        
        # Stop semantic memory indexing thread
        if self.semantic_memory and hasattr(self.semantic_memory, 'vector_store'):
            try:
                vector_store = self.semantic_memory.vector_store
                if hasattr(vector_store, '_background_thread') and vector_store._background_thread:
                    # Set stop flag if available
                    if hasattr(vector_store, '_should_stop'):
                        vector_store._should_stop = True
                    # Wait for thread to finish (with timeout)
                    if vector_store._background_thread.is_alive():
                        vector_store._background_thread.join(timeout=2)
                        if vector_store._background_thread.is_alive():
                            logger.warning("Memory indexing thread didn't stop within timeout")
                logger.debug("Semantic memory indexing stopped")
            except Exception as e:
                logger.warning(f"Failed to stop memory indexing: {e}")
        
        logger.info("Cleanup completed")
    
    async def run(self) -> None:
        """Start the async REPL loop."""
        # Enable royal blue terminal color
        enable_royal_blue_terminal()
        
        try:
            self.print_banner()
            
            # Initialize tab completion (if available)
            use_tab_completion = True
            completer = None
            history = None
            try:
                from intellishell.utils.completion import IntelliShellCompleter
                from prompt_toolkit import prompt as pt_prompt
                from prompt_toolkit.history import InMemoryHistory
                
                completer = IntelliShellCompleter(
                    provider_registry=self.registry,
                    parser=self.parser,
                    enable_smart_features=True  # Enable smart autocomplete
                )
                self._completer = completer  # Store for command recording
                history = InMemoryHistory()
                logger.info("Smart autocomplete initialized")
            except ImportError:
                use_tab_completion = False
                logger.warning("prompt-toolkit not available, tab completion disabled")
            except Exception as e:
                use_tab_completion = False
                logger.warning(f"Failed to initialize tab completion: {e}")
            
            while self.running:
                try:
                    # Use prompt_toolkit for tab completion if available
                    if use_tab_completion and completer:
                        from prompt_toolkit import prompt as pt_prompt
                        from prompt_toolkit.formatted_text import ANSI
                        
                        # For prompt_toolkit, use ANSI wrapper for colored prompt
                        if TerminalColors.supports_color():
                            colored_prompt = ANSI(TerminalColors.colorize(self.prompt, TerminalColors.ROYAL_BLUE))
                        else:
                            colored_prompt = self.prompt
                        
                        user_input = await asyncio.to_thread(
                            pt_prompt,
                            colored_prompt,
                            completer=completer,
                            history=history,
                            complete_while_typing=True,
                            enable_open_in_editor=True,
                        )
                    else:
                        # Fallback to standard input with colored prompt
                        colored_prompt = TerminalColors.colorize(self.prompt, TerminalColors.ROYAL_BLUE) if TerminalColors.supports_color() else self.prompt
                        user_input = await asyncio.to_thread(input, colored_prompt)
                    
                    self.running = await self.process_command(user_input)
                except KeyboardInterrupt:
                    print("\nUse 'exit' to quit.")
                    continue
                except EOFError:
                    exit_msg = format_message("\nExiting IntelliShell...", success=True)
                    print(exit_msg)
                    break
                except Exception as e:
                    error_msg = format_message(f"Error: {e}", success=False, is_error=True)
                    print(error_msg)
                    logger.exception("REPL error")
                    continue
            
            if self.debug:
                print()
                self._show_stats()
        finally:
            # Clean up all resources
            self.cleanup()
            # Reset terminal color on exit
            reset_terminal_color()


def main(
    debug: bool = False,
    enable_ai: bool = True,
    enable_memory: bool = True,
    enable_self_healing: bool = True
) -> None:
    """Entry point for IntelliShell REPL."""
    shell = IntentShell(
        debug=debug,
        enable_ai=enable_ai,
        enable_memory=enable_memory,
        enable_self_healing=enable_self_healing
    )
    asyncio.run(shell.run())


if __name__ == "__main__":
    main()
