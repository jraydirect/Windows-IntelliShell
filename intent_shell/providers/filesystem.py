"""Filesystem provider for directory and file operations."""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from intent_shell.providers.base import (
    BaseProvider,
    IntentTrigger,
    ExecutionResult,
    ProviderCapability
)


def _resolve_special_folder(folder_name: str) -> Optional[Path]:
    """
    Resolve Windows special folder path, handling OneDrive redirection.
    
    Args:
        folder_name: Name of folder (Desktop, Downloads, Documents, etc.)
        
    Returns:
        Path to folder or None if not found
    """
    # Try standard user profile location
    standard_path = Path.home() / folder_name
    if standard_path.exists():
        return standard_path
    
    # Try OneDrive location (common on Windows with OneDrive enabled)
    onedrive_path = Path.home() / "OneDrive" / folder_name
    if onedrive_path.exists():
        return onedrive_path
    
    # Try environment variables for known folders
    if folder_name == "Desktop":
        desktop_env = os.environ.get("USERPROFILE", "")
        if desktop_env:
            desktop_path = Path(desktop_env) / "Desktop"
            if desktop_path.exists():
                return desktop_path
            # Try OneDrive via USERPROFILE
            onedrive_desktop = Path(desktop_env) / "OneDrive" / "Desktop"
            if onedrive_desktop.exists():
                return onedrive_desktop
    
    if folder_name == "Downloads":
        downloads_env = os.environ.get("USERPROFILE", "")
        if downloads_env:
            downloads_path = Path(downloads_env) / "Downloads"
            if downloads_path.exists():
                return downloads_path
            # Try OneDrive via USERPROFILE
            onedrive_downloads = Path(downloads_env) / "OneDrive" / "Downloads"
            if onedrive_downloads.exists():
                return onedrive_downloads
    
    # Fallback to standard location even if it doesn't exist
    return standard_path


class FileSystemProvider(BaseProvider):
    """Provider for filesystem operations (read-only)."""
    
    @property
    def name(self) -> str:
        return "filesystem"
    
    @property
    def description(self) -> str:
        return "Filesystem navigation and directory access"
    
    def _initialize_triggers(self) -> None:
        """Initialize filesystem-related triggers."""
        self.capabilities = [
            ProviderCapability.READ_ONLY,
            ProviderCapability.ASYNC,
        ]
        
        self.triggers = [
            IntentTrigger(
                pattern="open desktop",
                intent_name="open_desktop",
                weight=1.0,
                aliases=["show desktop", "desktop folder", "go to desktop"]
            ),
            IntentTrigger(
                pattern="open downloads",
                intent_name="open_downloads",
                weight=1.0,
                aliases=["show downloads", "downloads folder", "go to downloads"]
            ),
            IntentTrigger(
                pattern="open documents",
                intent_name="open_documents",
                weight=1.0,
                aliases=["show documents", "documents folder", "my documents"]
            ),
            IntentTrigger(
                pattern="open recycle bin",
                intent_name="open_recycle_bin",
                weight=1.0,
                aliases=["show recycle bin", "trash", "recycle bin"]
            ),
            IntentTrigger(
                pattern="open explorer",
                intent_name="open_explorer",
                weight=1.0,
                aliases=["file explorer", "show explorer"]
            ),
            IntentTrigger(
                pattern="open home",
                intent_name="open_home",
                weight=1.0,
                aliases=["home folder", "user folder"]
            ),
            IntentTrigger(
                pattern="list files",
                intent_name="list_files",
                weight=0.9,
                aliases=[
                    "show files",
                    "files in",
                    "recent files",
                    "what files",
                    "what are the files",
                    "show me files"
                ]
            ),
            IntentTrigger(
                pattern="list downloads",
                intent_name="list_downloads",
                weight=1.0,
                aliases=[
                    "show downloads files",
                    "downloads files",
                    "recent downloads",
                    "files in downloads",
                    "what's in downloads",
                    "recent files in downloads",
                    "files in download folder",
                    "recent files in download",
                ]
            ),
            IntentTrigger(
                pattern="list desktop",
                intent_name="list_desktop",
                weight=1.0,
                aliases=[
                    "show desktop files",
                    "files on desktop",
                    "recent desktop files"
                ]
            ),
        ]
    
    async def execute(
        self,
        intent_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute filesystem operation."""
        try:
            if intent_name == "open_desktop":
                return await self._open_desktop()
            elif intent_name == "open_downloads":
                return await self._open_downloads()
            elif intent_name == "open_documents":
                return await self._open_documents()
            elif intent_name == "open_recycle_bin":
                return await self._open_recycle_bin()
            elif intent_name == "open_explorer":
                return await self._open_explorer()
            elif intent_name == "open_home":
                return await self._open_home()
            elif intent_name == "list_files":
                return await self._list_files(context)
            elif intent_name == "list_downloads":
                return await self._list_downloads()
            elif intent_name == "list_desktop":
                return await self._list_desktop()
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
    
    async def _open_desktop(self) -> ExecutionResult:
        """Open Desktop folder."""
        desktop_path = _resolve_special_folder("Desktop")
        if not desktop_path or not desktop_path.exists():
            return ExecutionResult(
                success=False,
                message=f"Desktop path not found: {desktop_path}"
            )
        os.startfile(desktop_path)
        return ExecutionResult(
            success=True,
            message=f"Opening Desktop: {desktop_path}",
            data={"path": str(desktop_path)}
        )
    
    async def _open_downloads(self) -> ExecutionResult:
        """Open Downloads folder."""
        downloads_path = _resolve_special_folder("Downloads")
        if not downloads_path or not downloads_path.exists():
            return ExecutionResult(
                success=False,
                message=f"Downloads path not found: {downloads_path}"
            )
        os.startfile(downloads_path)
        return ExecutionResult(
            success=True,
            message=f"Opening Downloads: {downloads_path}",
            data={"path": str(downloads_path)}
        )
    
    async def _open_documents(self) -> ExecutionResult:
        """Open Documents folder."""
        documents_path = _resolve_special_folder("Documents")
        if not documents_path or not documents_path.exists():
            return ExecutionResult(
                success=False,
                message=f"Documents path not found: {documents_path}"
            )
        os.startfile(documents_path)
        return ExecutionResult(
            success=True,
            message=f"Opening Documents: {documents_path}",
            data={"path": str(documents_path)}
        )
    
    async def _open_recycle_bin(self) -> ExecutionResult:
        """Open Recycle Bin."""
        import subprocess
        subprocess.run(["explorer", "shell:RecycleBinFolder"], check=True)
        return ExecutionResult(
            success=True,
            message="Opening Recycle Bin...",
            data={"shell_path": "shell:RecycleBinFolder"}
        )
    
    async def _open_explorer(self) -> ExecutionResult:
        """Open File Explorer."""
        os.startfile("explorer")
        return ExecutionResult(
            success=True,
            message="Opening File Explorer..."
        )
    
    async def _open_home(self) -> ExecutionResult:
        """Open home directory."""
        home_path = Path.home()
        os.startfile(home_path)
        return ExecutionResult(
            success=True,
            message=f"Opening Home: {home_path}",
            data={"path": str(home_path)}
        )
    
    async def _list_files(self, context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """List files in a directory."""
        # Try to extract directory from context
        target_dir = None
        
        if context and "original_input" in context:
            input_lower = context["original_input"].lower()
            
            # Check for common folders (with OneDrive support)
            if "download" in input_lower:
                target_dir = _resolve_special_folder("Downloads")
            elif "desktop" in input_lower:
                target_dir = _resolve_special_folder("Desktop")
            elif "document" in input_lower:
                target_dir = _resolve_special_folder("Documents")
        
        # Default to downloads
        if target_dir is None:
            target_dir = _resolve_special_folder("Downloads")
        
        if not target_dir.exists():
            return ExecutionResult(
                success=False,
                message=f"Directory not found: {target_dir}"
            )
        
        try:
            # Get files sorted by modification time
            files = []
            for item in target_dir.iterdir():
                if item.is_file():
                    stat = item.stat()
                    files.append({
                        "name": item.name,
                        "size_mb": stat.st_size / (1024 * 1024),
                        "modified": stat.st_mtime
                    })
            
            # Sort by modified time (newest first)
            files.sort(key=lambda f: f["modified"], reverse=True)
            
            # Take top 10
            recent_files = files[:10]
            
            if not recent_files:
                from intent_shell.utils.display import format_message
                return ExecutionResult(
                    success=True,
                    message=format_message(f"No files found in {target_dir}", success=True)
                )
            
            # Format as rich table
            from intent_shell.utils.display import format_file_table
            title = f"Recent files in {target_dir.name}"
            formatted_table = format_file_table(recent_files, title)
            
            return ExecutionResult(
                success=True,
                message=formatted_table,
                data={"files": recent_files, "directory": str(target_dir)}
            )
            
        except PermissionError:
            return ExecutionResult(
                success=False,
                message=f"Permission denied accessing {target_dir}"
            )
    
    async def _list_downloads(self) -> ExecutionResult:
        """List files in Downloads folder."""
        return await self._list_files({"original_input": "list downloads"})
    
    async def _list_desktop(self) -> ExecutionResult:
        """List files in Desktop folder."""
        return await self._list_files({"original_input": "list desktop"})
