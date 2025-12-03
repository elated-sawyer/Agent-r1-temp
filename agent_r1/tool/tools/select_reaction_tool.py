"""
Single step retro tool implementation
"""

from typing import Dict
import json
from agent_r1.tool.tool_base import Tool


class SelectReactionTool(Tool):
    """
    Tool for performing Single step retro
    """
    
    def __init__(self):
        """
        Initialize the calculator tool
        """
        name = "select_reaction"
        description = "Given several reactions to synthesize a molecule, use this tool to select one from them."
        parameters = {
            "type": "object",
            "properties": {
                "reaction": {
                    "type": "string",
                    "description": "The ID for the selected reaction. For example, 0-0-0 ."
                }
            },
            "required": ["reaction"]
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
            return 0.0 # -0.1  # 无法解析结果

    
if __name__ == "__main__":
    calculator_tool = CalculatorTool()
    print(calculator_tool.execute({"expression": "2 + 3"}))
    print(calculator_tool.calculate_reward({"expression": "2 + 3"}, "Result: 5"))
