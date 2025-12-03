"""
Specific tool implementations
"""

from agent_r1.tool.tools.search_tool import SearchTool
from agent_r1.tool.tools.calculator_tool import CalculatorTool
from agent_r1.tool.tools.wiki_search_tool import WikiSearchTool
from agent_r1.tool.tools.single_step_retro_tool import SingleStepRetroTool
from agent_r1.tool.tools.select_reaction_tool import SelectReactionTool
from agent_r1.tool.tools.back_state_tool import BackStateTool



__all__ = [
    'SearchTool',
    'CalculatorTool',
    'WikiSearchTool',
    'SingleStepRetroTool',
    'SelectReactionTool',
    'BackStateTool',
] 

def _default_tools(env):
    if env == 'search':
        return [SearchTool()]
    elif env == 'calculator':
        return [CalculatorTool()]
    elif env == 'wikisearch':
        return [WikiSearchTool()]
    elif env == 'retro':
        return [SingleStepRetroTool(), SelectReactionTool(), BackStateTool()]
    elif env == 'retro_noback':
        return [SingleStepRetroTool(), SelectReactionTool()]
    elif env == 'retro_noback_V4':
        return [SingleStepRetroTool(mlp_model_dump='./one_step_model/retro_star_V4.ckpt'), SelectReactionTool()]
    elif env == 'retro_noback_V2':
        return [SingleStepRetroTool(mlp_model_dump='./one_step_model/retro_star_value_ours.ckpt'), SelectReactionTool()]
    else:
        raise NotImplementedError
