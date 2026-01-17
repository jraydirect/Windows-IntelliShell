"""Application provider for launching Windows applications."""

import os
import re
import subprocess
from typing import Optional, Dict, Any
from intellishell.providers.base import (
    BaseProvider,
    IntentTrigger,
    ExecutionResult,
    ProviderCapability
)


class AppProvider(BaseProvider):
    """Provider for launching Windows applications."""
    
    @property
    def name(self) -> str:
        return "app"
    
    @property
    def description(self) -> str:
        return "Application launcher for Windows programs"
    
    def _initialize_triggers(self) -> None:
        """Initialize app-related triggers."""
        self.capabilities = [
            ProviderCapability.READ_ONLY,
            ProviderCapability.ASYNC,
        ]
        
        self.triggers = [
            IntentTrigger(
                pattern="open notepad",
                intent_name="launch_notepad",
                weight=1.0,
                aliases=["start notepad", "notepad", "text editor"]
            ),
            IntentTrigger(
                pattern="open calculator",
                intent_name="launch_calculator",
                weight=1.0,
                aliases=["start calculator", "calculator", "calc"]
            ),
            IntentTrigger(
                pattern="open settings",
                intent_name="launch_settings",
                weight=1.0,
                aliases=["windows settings", "system settings", "settings"]
            ),
            IntentTrigger(
                pattern="open task manager",
                intent_name="launch_task_manager",
                weight=1.0,
                aliases=["task manager", "taskmgr"]
            ),
            IntentTrigger(
                pattern="open control panel",
                intent_name="launch_control_panel",
                weight=1.0,
                aliases=["control panel", "control"]
            ),
            IntentTrigger(
                pattern="open startup folder",
                intent_name="open_startup",
                weight=1.0,
                aliases=["startup folder", "startup apps"]
            ),
            IntentTrigger(
                pattern="open app",
                intent_name="launch_app",
                weight=0.9,
                aliases=["launch app", "start app", "run app", "open"]
            ),
        ]
    
    async def execute(
        self,
        intent_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute application launch."""
        try:
            if intent_name == "launch_notepad":
                return await self._launch_notepad()
            elif intent_name == "launch_calculator":
                return await self._launch_calculator()
            elif intent_name == "launch_settings":
                return await self._launch_settings()
            elif intent_name == "launch_task_manager":
                return await self._launch_task_manager()
            elif intent_name == "launch_control_panel":
                return await self._launch_control_panel()
            elif intent_name == "open_startup":
                return await self._open_startup()
            elif intent_name == "launch_app":
                return await self._launch_app(context)
            else:
                return ExecutionResult(
                    success=False,
                    message=f"Unknown intent: {intent_name}"
                )
        except Exception as e:
            return ExecutionResult(
                success=False,
                message=f"Execution error: {e}"
            )
    
    async def _launch_notepad(self) -> ExecutionResult:
        """Launch Notepad."""
        os.startfile("notepad")
        return ExecutionResult(
            success=True,
            message="Opening Notepad...",
            data={"app": "notepad"}
        )
    
    async def _launch_calculator(self) -> ExecutionResult:
        """Launch Calculator."""
        os.startfile("calc")
        return ExecutionResult(
            success=True,
            message="Opening Calculator...",
            data={"app": "calc"}
        )
    
    async def _launch_settings(self) -> ExecutionResult:
        """Launch Windows Settings."""
        os.startfile("ms-settings:")
        return ExecutionResult(
            success=True,
            message="Opening Windows Settings...",
            data={"app": "ms-settings:"}
        )
    
    async def _launch_task_manager(self) -> ExecutionResult:
        """Launch Task Manager."""
        subprocess.Popen(["taskmgr"])
        return ExecutionResult(
            success=True,
            message="Opening Task Manager...",
            data={"app": "taskmgr"}
        )
    
    async def _launch_control_panel(self) -> ExecutionResult:
        """Launch Control Panel."""
        os.startfile("control")
        return ExecutionResult(
            success=True,
            message="Opening Control Panel...",
            data={"app": "control"}
        )
    
    async def _open_startup(self) -> ExecutionResult:
        """Open Startup folder."""
        subprocess.run(["explorer", "shell:Startup"], check=True)
        return ExecutionResult(
            success=True,
            message="Opening Startup folder...",
            data={"shell_path": "shell:Startup"}
        )
    
    async def _launch_app(self, context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """Launch an application by name."""
        if not context or "original_input" not in context:
            return ExecutionResult(
                success=False,
                message="No application name specified. Usage: open brave"
            )
        
        # First, try to get app name from LLM parameters
        app_name = None
        if context.get("parameters") and "app_name" in context["parameters"]:
            app_name = context["parameters"]["app_name"].lower().strip()
        elif context.get("parameters") and "app" in context["parameters"]:
            app_name = context["parameters"]["app"].lower().strip()
        
        # If not in parameters, extract from original input
        if not app_name:
            input_text = context["original_input"].lower().strip()
            
            # Extract app name from natural language patterns
            # Handle patterns like: "open brave", "launch brave", "can you open brave", "open the app brave", etc.
            # Try to find app name after common verbs and phrases
            patterns = [
                r"(?:open|launch|start|run)\s+(?:the\s+)?(?:app\s+)?(\w+)",
                r"(?:can\s+you\s+|please\s+)?(?:open|launch|start|run)\s+(?:the\s+)?(?:app\s+)?(\w+)",
                r"(\w+)\s+(?:app|application)",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, input_text)
                if match:
                    app_name = match.group(1).strip()
                    break
            
            # Fallback: remove common prefixes from start
            if not app_name:
                prefixes = ["open", "launch", "start", "run", "can you open", "please open"]
                app_name = input_text
                for prefix in sorted(prefixes, key=len, reverse=True):  # Try longest first
                    if app_name.startswith(prefix):
                        app_name = app_name[len(prefix):].strip()
                        # Remove "the" and "app" if they follow
                        app_name = re.sub(r"^(the\s+)?(app\s+)?", "", app_name).strip()
                        break
        
        if not app_name:
            return ExecutionResult(
                success=False,
                message="No application name specified. Usage: open brave"
            )
        
        # Map common app names to their executable names
        app_mapping = {
            "brave": "brave.exe",
            "chrome": "chrome.exe",
            "firefox": "firefox.exe",
            "edge": "msedge.exe",
            "discord": "discord.exe",
            "notepad": "notepad.exe",
            "cursor": "cursor.exe",
            "code": "code.exe",
            "vscode": "code.exe",
            "visual studio code": "code.exe",
            "spotify": "spotify.exe",
            "steam": "steam.exe",
            "vlc": "vlc.exe",
            "photoshop": "photoshop.exe",
            "paint": "mspaint.exe",
            "word": "winword.exe",
            "excel": "excel.exe",
            "powerpoint": "powerpnt.exe",
            "outlook": "outlook.exe",
        }
        
        # Try mapped name first
        executable = app_mapping.get(app_name, None)
        
        # If not in mapping, try the name as-is (Windows might find it)
        if not executable:
            # Try with .exe extension
            if not app_name.endswith(".exe"):
                executable = f"{app_name}.exe"
            else:
                executable = app_name
        
        try:
            # Try launching via os.startfile (Windows will search PATH and Start Menu)
            # This works for most installed applications
            os.startfile(executable)
            
            # Also try subprocess as fallback for some apps
            # But startfile is usually better for Windows apps
            return ExecutionResult(
                success=True,
                message=f"Opening {app_name.title()}...",
                data={"app": executable}
            )
        except FileNotFoundError:
            # If startfile fails, try subprocess with shell=True (searches PATH)
            try:
                subprocess.Popen([executable], shell=True)
                return ExecutionResult(
                    success=True,
                    message=f"Opening {app_name.title()}...",
                    data={"app": executable}
                )
            except Exception as e:
                return ExecutionResult(
                    success=False,
                    message=f"Could not launch '{app_name}'. Make sure it's installed and in your PATH, or try the full path."
                )
        except Exception as e:
            return ExecutionResult(
                success=False,
                message=f"Error launching '{app_name}': {e}"
            )