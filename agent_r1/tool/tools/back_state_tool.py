"""
Single step retro tool implementation
"""

from typing import Dict
import json
from agent_r1.tool.tool_base import Tool


class BackStateTool(Tool):
    """
    Tool for performing Single step retro
    """
    
    def __init__(self):
        """
        Initialize the calculator tool
        """
        name = "back_state"
        description = "Use this tool to go to a history state when necessary."
        parameters = {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "description": "The ID for the selected history state, which can be a molecule state or a reaction state. For example,  0 or 0-0 ."
                }
            },
            "required": ["state"]
        }

        
        super().__init__(name, description, parameters)
    
    def execute(self, args: Dict) -> str:
        """
        Execute calculator operations
        
        Args:
            args: Tool parameters, containing:
                - "expression": arithmetic expression to evaluate
            
        Returns:
            Result of the calculation
        """

        try:
            return {"results": ""}
        
        except Exception as e:
            return {"error": str(e)}
    
    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        TODO: implement the reward for molecule selection
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value
        """
        try:
            # result_obj = json.loads(result)
            result_obj = result
            # 有效的工具调用
            if "results" in result_obj:
                return 0.0
            elif "error" in result_obj:
                return 0.0 #-0.1  # 轻微惩罚错误
            else:
                return 0.0
        except:
            return 0.0 #-0.1  # 无法解析结果

    
if __name__ == "__main__":
    calculator_tool = CalculatorTool()
    print(calculator_tool.execute({"expression": "2 + 3"}))
    print(calculator_tool.calculate_reward({"expression": "2 + 3"}, "Result: 5"))
