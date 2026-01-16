"""Local LLM bridge for agentic reasoning."""

import json
import logging
import requests
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ValidationError
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class IntentRequest(BaseModel):
    """Pydantic model for LLM intent requests."""
    intent: str = Field(..., description="The intent to execute")
    provider: str = Field(..., description="Provider name to use")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Intent parameters")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    reasoning: Optional[str] = Field(None, description="LLM reasoning for decision")


class LLMResponse(BaseModel):
    """Pydantic model for LLM responses."""
    success: bool = Field(..., description="Whether LLM processing succeeded")
    intent_request: Optional[IntentRequest] = Field(None, description="Parsed intent request")
    message: Optional[str] = Field(None, description="Response message")
    raw_response: Optional[str] = Field(None, description="Raw LLM output")


@dataclass
class ShortTermMemory:
    """Short-term memory context for LLM."""
    user_input: str
    intent_name: Optional[str]
    result: Optional[str]
    timestamp: str
    entities: List[Dict] = None
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []


class ContextManager:
    """
    Manages short-term context for LLM interactions.
    
    Keeps the last N interactions in RAM for pronoun resolution
    and contextual understanding.
    """
    
    def __init__(self, max_history: int = 5):
        """
        Initialize context manager.
        
        Args:
            max_history: Maximum number of interactions to keep
        """
        self.max_history = max_history
        self.history: List[ShortTermMemory] = []
        self.clipboard_content: Optional[str] = None
        self.last_directory: Optional[str] = None
    
    def add_interaction(
        self,
        user_input: str,
        intent_name: Optional[str],
        result: Optional[str],
        entities: Optional[List[Dict]] = None
    ) -> None:
        """
        Add an interaction to context.
        
        Args:
            user_input: User input
            intent_name: Matched intent
            result: Result message
            entities: Extracted entities
        """
        from datetime import datetime
        
        memory = ShortTermMemory(
            user_input=user_input,
            intent_name=intent_name,
            result=result,
            timestamp=datetime.now().isoformat(),
            entities=entities or []
        )
        
        self.history.append(memory)
        
        # Keep only last N
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_context_prompt(self) -> str:
        """
        Generate context prompt for LLM.
        
        Returns:
            Formatted context string
        """
        if not self.history:
            return "No recent context available."
        
        context_parts = ["Recent interactions:"]
        
        for i, mem in enumerate(self.history, 1):
            context_parts.append(f"{i}. User: {mem.user_input}")
            if mem.intent_name:
                context_parts.append(f"   Intent: {mem.intent_name}")
            if mem.result:
                # Truncate long results
                result_preview = mem.result[:100] + "..." if len(mem.result) > 100 else mem.result
                context_parts.append(f"   Result: {result_preview}")
        
        if self.clipboard_content:
            context_parts.append(f"\nClipboard: {self.clipboard_content[:100]}")
        
        if self.last_directory:
            context_parts.append(f"Last directory: {self.last_directory}")
        
        return "\n".join(context_parts)
    
    def update_clipboard(self, content: str) -> None:
        """Update clipboard content in context."""
        self.clipboard_content = content
    
    def update_directory(self, directory: str) -> None:
        """Update last accessed directory."""
        self.last_directory = directory


class OllamaClient:
    """
    Client for local Ollama LLM API.
    
    Provides tool-use capabilities for IntelliShell.
    """
    
    DEFAULT_HOST = "http://localhost:11434"
    DEFAULT_MODEL = "llama3:8b"
    
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        timeout: int = 120  # Increased for larger prompts and slower models
    ):
        """
        Initialize Ollama client.
        
        Args:
            host: Ollama API host
            model: Model name to use
            timeout: Request timeout in seconds
        """
        self.host = host
        self.model = model
        self.timeout = timeout
        self._available = None
    
    def is_available(self) -> bool:
        """
        Check if Ollama is running and available.
        
        Returns:
            True if available, False otherwise
        """
        if self._available is not None:
            return self._available
        
        try:
            response = requests.get(
                f"{self.host}/api/tags",
                timeout=2
            )
            
            if response.status_code == 200:
                # Check if model exists
                try:
                    data = response.json()
                    models = data.get("models", [])
                    model_names = [m.get("name", "") for m in models]
                    
                    # Check if our model is available
                    if not any(self.model in name for name in model_names):
                        logger.warning(
                            f"Ollama running but model '{self.model}' not found. "
                            f"Available models: {model_names}. "
                            f"Run: ollama pull {self.model}"
                        )
                        self._available = False
                        return False
                except:
                    pass
                
                self._available = True
                logger.info(f"Ollama available at {self.host} with model {self.model}")
                return True
            else:
                logger.warning(f"Ollama responded with status {response.status_code}")
                self._available = False
                return False
            
        except requests.exceptions.RequestException as e:
            logger.debug(f"Ollama not available: {e}")
            self._available = False
            return False
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1
    ) -> Optional[str]:
        """
        Generate completion from Ollama.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Temperature for generation
            
        Returns:
            Generated text or None if failed
        """
        if not self.is_available():
            logger.warning("Ollama not available for generation")
            return None
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 128  # Reduced - we only need JSON, not long text
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"Ollama generation failed: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timed out after {self.timeout}s. Model may be too slow or prompt too large.")
            return None
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return None


class AIBridge:
    """
    Bridge between IntelliShell and local LLM.
    
    Provides agentic reasoning and fallback for low-confidence matches.
    """
    
    def _build_system_prompt(self) -> str:
        """
        Build system prompt dynamically from provider registry.
        
        Returns:
            System prompt string with all available tools/intents
        """
        if self._system_prompt is not None:
            return self._system_prompt
        
        if not self.provider_registry:
            # Fallback to static prompt if no registry
            return self._get_static_system_prompt()
        
        # Build concise dynamic prompt from registry
        prompt_parts = [
            "You are an AI assistant for IntelliShell, a Windows command automation tool.",
            "Map user commands to structured intents (tool calls).",
            "",
            "Available Tools:"
        ]
        
        # Build compact intent list per provider
        providers = self.provider_registry.get_all_providers()
        for provider in providers:
            intents = sorted(set(trigger.intent_name for trigger in provider.get_triggers()))
            if intents:
                # Compact format: provider: intent1, intent2, ...
                prompt_parts.append(f"- {provider.name}: {', '.join(intents)}")
        
        prompt_parts.extend([
            "",
            "Special Commands (handled internally, NOT provider intents):",
            "- 'help' / '?' - Show available commands (handled internally, NOT a provider intent)",
            "- 'manifest' - Show system manifest",
            "- 'history' - Show command history",
            "- 'stats' - Show session statistics",
            "",
            "Output ONLY valid JSON:",
            '{"intent": "intent_name", "provider": "provider_name", "parameters": {}, "confidence": 0.0-1.0, "reasoning": "brief"}',
            "",
            "Rules:",
            "- For 'what are available commands' / 'show commands' / 'list commands' → suggest user type 'help'",
            "- For 'what are my recent commands' / 'show my history' / 'what did I do' → suggest user type 'history' (handled internally)",
            "- For 'what files in X' or 'show files in X' → use list_downloads/list_desktop/list_files",
            "- For 'what's in downloads' → list_downloads (not open_downloads)",
            "- For 'what processes' / 'show processes' / 'list processes' → use list_processes (READ-ONLY)",
            "- For 'open [app name]' like 'open brave', 'open discord', 'open chrome' → use launch_app (app provider)",
            "- For 'open [folder]' like 'open desktop', 'open downloads' → use open_desktop/open_downloads (filesystem provider)",
            "- Questions about contents → list_* intents, not open_*",
            "- Questions asking 'what' / 'show' / 'list' → use READ-ONLY list_* intents, NOT kill/delete intents",
            "- NEVER route viewing/listing queries to destructive actions (kill, delete, etc.)",
            "- DO NOT create intents that don't exist! Only use intents from the provider list above.",
            "- DO NOT route help-related queries to providers. Help is handled internally before routing.",
            "",
            "Examples:",
            '"what files are in downloads?" → {"intent": "list_downloads", "provider": "filesystem", "confidence": 0.95, "reasoning": "list files"}',
            '"what processes are running?" → {"intent": "list_processes", "provider": "system", "confidence": 0.95, "reasoning": "list processes"}',
            '"show me running processes" → {"intent": "list_processes", "provider": "system", "confidence": 0.95, "reasoning": "list processes"}',
            '"open brave" → {"intent": "launch_app", "provider": "app", "confidence": 0.95, "reasoning": "launch application"}',
            '"open discord" → {"intent": "launch_app", "provider": "app", "confidence": 0.95, "reasoning": "launch application"}',
            '"what\'s my computer name?" → {"intent": "get_hostname", "provider": "system_monitor", "confidence": 0.95, "reasoning": "get hostname"}',
        ])
        
        self._system_prompt = "\n".join(prompt_parts)
        return self._system_prompt
    
    @staticmethod
    def _get_static_system_prompt() -> str:
        """Fallback static system prompt if no registry available."""
        return """You are an AI assistant for IntelliShell, a Windows command automation tool.

Your job is to interpret user commands and map them to structured intents.

Available Providers and Intents:
- filesystem: open_desktop, open_downloads, open_documents, open_home, open_recycle_bin, open_explorer, list_files, list_downloads, list_desktop
- app: launch_notepad, launch_calculator, launch_settings, launch_task_manager, launch_control_panel, open_startup, launch_app
- system_monitor: get_system_info, get_hostname, get_username, get_disk_space
- system: list_processes, kill_process, kill_by_name, kill_most_memory, check_admin
- watch: watch_downloads, watch_for_file_type, stop_watch, list_watches
- memory: recall_command, recall_folder, show_recent
- doctor: system_health, check_deps

Output ONLY valid JSON in this format:
{
  "intent": "intent_name",
  "provider": "provider_name",
  "parameters": {},
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}

When user asks about files/folders, use list_files or list_downloads/list_desktop, not open_* intents.
"""
    
    def __init__(
        self,
        ollama_host: str = OllamaClient.DEFAULT_HOST,
        ollama_model: str = OllamaClient.DEFAULT_MODEL,
        provider_registry=None
    ):
        """
        Initialize AI bridge.
        
        Args:
            ollama_host: Ollama API host
            ollama_model: Model name
            provider_registry: Optional ProviderRegistry for dynamic tool mapping
        """
        self.ollama = OllamaClient(host=ollama_host, model=ollama_model)
        self.context_manager = ContextManager()
        self.provider_registry = provider_registry
        self._system_prompt = None
    
    def is_available(self) -> bool:
        """Check if AI bridge is available."""
        return self.ollama.is_available()
    
    def interpret_command(
        self,
        user_input: str,
        include_context: bool = True
    ) -> LLMResponse:
        """
        Interpret user command using LLM.
        
        Args:
            user_input: User input to interpret
            include_context: Include short-term context
            
        Returns:
            LLMResponse with parsed intent
        """
        if not self.is_available():
            return LLMResponse(
                success=False,
                message="LLM not available"
            )
        
        # Build prompt
        prompt_parts = []
        
        if include_context:
            context = self.context_manager.get_context_prompt()
            prompt_parts.append(context)
            prompt_parts.append("")
        
        prompt_parts.append(f"User: {user_input}")
        prompt_parts.append("\nOutput JSON:")
        
        prompt = "\n".join(prompt_parts)
        
        # Generate
        try:
            system_prompt = self._build_system_prompt()
            raw_response = self.ollama.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1
            )
            
            if not raw_response:
                logger.warning("LLM returned empty response - may have timed out or model too slow")
                return LLMResponse(
                    success=False,
                    message="LLM request timed out or returned empty response. Try a smaller model or increase timeout."
                )
            
            # Parse JSON
            # Extract JSON from response (may have extra text)
            json_str = self._extract_json(raw_response)
            
            if not json_str:
                return LLMResponse(
                    success=False,
                    message="Could not extract JSON from LLM response",
                    raw_response=raw_response
                )
            
            # Validate with Pydantic
            intent_data = json.loads(json_str)
            intent_request = IntentRequest(**intent_data)
            
            return LLMResponse(
                success=True,
                intent_request=intent_request,
                raw_response=raw_response
            )
            
        except ValidationError as e:
            logger.error(f"LLM response validation failed: {e}")
            return LLMResponse(
                success=False,
                message=f"Invalid LLM response format: {e}",
                raw_response=raw_response if 'raw_response' in locals() else None
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON: {e}")
            return LLMResponse(
                success=False,
                message=f"Invalid JSON from LLM: {e}",
                raw_response=raw_response if 'raw_response' in locals() else None
            )
        except Exception as e:
            logger.error(f"LLM interpretation error: {e}")
            return LLMResponse(
                success=False,
                message=f"LLM error: {e}"
            )
    
    def _extract_json(self, text: str) -> Optional[str]:
        """
        Extract JSON from text that may contain other content.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Extracted JSON string or None
        """
        # Find JSON object boundaries
        start = text.find("{")
        end = text.rfind("}") + 1
        
        if start >= 0 and end > start:
            return text[start:end]
        
        return None
    
    def add_context(
        self,
        user_input: str,
        intent_name: Optional[str],
        result: Optional[str],
        entities: Optional[List[Dict]] = None
    ) -> None:
        """Add interaction to context manager."""
        self.context_manager.add_interaction(
            user_input, intent_name, result, entities
        )
