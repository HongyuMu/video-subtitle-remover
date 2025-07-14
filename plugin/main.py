from dify_plugin import Plugin, DifyPluginEnv
from provider.validation import SubRemoverProvider
from tools.sub_remover import SubRemoverTool
import logging

# Initialize the plugin with logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

plugin = Plugin(DifyPluginEnv(MAX_REQUEST_TIMEOUT=120))

try:
    # Add the tool and provider
    plugin.add_tool_provider(SubRemoverProvider())
    plugin.add_tool(SubRemoverTool())
    logging.info("Plugin initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing plugin: {e}")
    raise

if __name__ == "__main__":
    plugin.run()
