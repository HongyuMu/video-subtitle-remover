from dify_plugin import Plugin, DifyPluginEnv
from provider.validation import SubRemoverProvider
from tools.sub_remover import SubRemoverTool

# Initialize the plugin
plugin = Plugin(DifyPluginEnv(MAX_REQUEST_TIMEOUT=120))

# Add the tool and provider
plugin.add_tool_provider(SubRemoverProvider())
plugin.add_tool(SubRemoverTool())

if __name__ == "__main__":
    plugin.run()
