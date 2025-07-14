from collections.abc import Generator
from typing import Any
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
import logging
from backend.main import SubtitleRemover

class SubRemoverTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        This method will be called when the tool is invoked.
        It will handle the subtitle removal logic.
        """
        video_path = tool_parameters.get("video_path")
        sub_area = tool_parameters.get("sub_area", None)
        
        if not video_path:
            raise ValueError("Video path is required")
        
        logging.info(f"Starting subtitle removal for video: {video_path}")
        
        try:
            # Set subtitle area if provided
            if sub_area:
                ymin, ymax, xmin, xmax = map(int, sub_area.split(","))
                sub_area_tuple = (ymin, ymax, xmin, xmax)
            else:
                sub_area_tuple = None
            
            # Run the subtitle removal using SubtitleRemover
            sd = SubtitleRemover(video_path, sub_area=sub_area_tuple)
            sd.run()  # Perform subtitle removal
            
            # If successful, return the result
            yield self.create_json_message({
                "result": f"Subtitle removal completed for {video_path}",
                "output_video_path": sd.video_out_name  # Path of the processed video
            })
        
        except FileNotFoundError as fnf_error:
            logging.error(f"File not found: {fnf_error}")
            yield self.create_json_message({
                "error": f"File not found: {str(fnf_error)}"
            })
        except ValueError as ve:
            logging.error(f"Value error: {ve}")
            yield self.create_json_message({
                "error": f"Invalid value: {str(ve)}"
            })
        
        except Exception as e:
            logging.error(f"Error during subtitle removal: {e}")
            yield self.create_json_message({
                "error": f"An error occurred during subtitle removal: {str(e)}"
            })
