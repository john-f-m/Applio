import gradio as gr
import sys
import os
import logging
from typing import Any, Optional, NoReturn
import socket

# Type alias for clarity
PortNumber = int
ServerName = str

# Constants with type annotations
DEFAULT_SERVER_NAME: ServerName = "127.0.0.1"
DEFAULT_PORT: PortNumber = 6969
MAX_PORT_ATTEMPTS: int = 10

# Set up logging
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add current directory to sys.path
now_dir: str = os.getcwd()
sys.path.append(now_dir)

# Zluda hijack
import rvc.lib.zluda

# Import Tabs
from tabs.inference.inference import inference_tab
from tabs.train.train import train_tab
from tabs.extra.extra import extra_tab
from tabs.report.report import report_tab
from tabs.download.download import download_tab
from tabs.tts.tts import tts_tab
from tabs.voice_blender.voice_blender import voice_blender_tab
from tabs.plugins.plugins import plugins_tab
from tabs.settings.settings import settings_tab

# Run prerequisites
from core import run_prerequisites_script

run_prerequisites_script(
    pretraineds_hifigan=True,
    models=True,
    exe=True,
)

# Initialize i18n
from rvc.lib.tools.i18n import I18nAuto

i18n: I18nAuto = I18nAuto()

# Gradio App
Applio: gr.Blocks = gr.Blocks(theme="ParityError/Anime", title="Applio")

# Callable type alias for tab functions
TabFunction = Any  # Since we don't know the exact return type of tab functions

with Applio:
    gr.Markdown("# Applio")
    gr.Markdown(
        i18n(
            "Ultimate voice cloning tool, meticulously optimized for unrivaled power, modularity, and user-friendly experience."
        )
    )
    gr.Markdown(
        i18n(
            "[Support](https://discord.gg/urxFjYmYYh) — [Docs](https://docs.applio.org/)"
        )
    )
    
    with gr.Tab(i18n("Inference")):
        inference_tab()

    with gr.Tab(i18n("Training")):
        train_tab()

    with gr.Tab(i18n("TTS")):
        tts_tab()

    with gr.Tab(i18n("Voice Blender")):
        voice_blender_tab()

    with gr.Tab(i18n("Plugins")):
        plugins_tab()

    with gr.Tab(i18n("Download")):
        download_tab()

    with gr.Tab(i18n("Report a Bug")):
        report_tab()

    with gr.Tab(i18n("Extra")):
        extra_tab()

    with gr.Tab(i18n("Settings")):
        settings_tab()

    gr.Markdown(
        """
    <div style="text-align: center; font-size: 0.9em; text-color: a3a3a3;">
    By using Applio, you agree to comply with ethical and legal standards, respect intellectual property and privacy rights, avoid harmful or prohibited uses, and accept full responsibility for any outcomes, while Applio disclaims liability and reserves the right to amend these terms.
    </div>
    """
    )


def launch_gradio(server_name: ServerName, server_port: PortNumber) -> None:
    """
    Launch the Gradio application with the specified server configuration.
    
    Args:
        server_name: The server hostname or IP address
        server_port: The port number to run the server on
    """
    Applio.launch(
        favicon_path="assets/ICON.ico",
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        server_name=server_name,
        server_port=server_port,
        quiet=True,
        prevent_thread_lock=True,
    )


def is_port_in_use(port: PortNumber) -> bool:
    """
    Check if a port is already in use.
    
    Args:
        port: The port number to check
        
    Returns:
        True if the port is in use, False otherwise
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def get_config() -> Optional[dict[str, Any]]:
    """
    Load configuration from config.json if it exists.
    
    Returns:
        Configuration dictionary or None if config file doesn't exist
    """
    if os.path.isfile("config.json"):
        with open("config.json", "r") as file:
            config: dict[str, Any] = json.load(file)
            return config
    return None


def main() -> NoReturn:
    """
    Main entry point for the application.
    Handles port allocation and launches the Gradio server.
    """
    port: PortNumber = DEFAULT_PORT
    
    # Get port from config if available
    config: Optional[dict[str, Any]] = get_config()
    if config and "server" in config and "port" in config["server"]:
        port = int(config["server"]["port"])
    
    # Override with command line argument if provided
    if "--port" in sys.argv:
        try:
            port_index: int = sys.argv.index("--port") + 1
            if port_index < len(sys.argv):
                port = int(sys.argv[port_index])
        except (ValueError, IndexError):
            print(f"Invalid port argument. Using default port {DEFAULT_PORT}")
            port = DEFAULT_PORT
    
    # Find available port
    original_port: PortNumber = port
    attempts: int = 0
    
    while is_port_in_use(port) and attempts < MAX_PORT_ATTEMPTS:
        print(f"Port {port} is in use. Trying port {port + 1}...")
        port += 1
        attempts += 1
    
    if attempts >= MAX_PORT_ATTEMPTS:
        print(f"Could not find an available port after {MAX_PORT_ATTEMPTS} attempts.")
        sys.exit(1)
    
    if port != original_port:
        print(f"Using port {port} instead of {original_port}")
    
    # Launch the application
    launch_gradio(DEFAULT_SERVER_NAME, port)


if __name__ == "__main__":
    import json
    main()
