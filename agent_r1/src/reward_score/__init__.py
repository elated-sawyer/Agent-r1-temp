def _default_compute_score_format(data_source, solution_str, extra_info=None):
    if data_source == 'hotpotqa/hotpot_qa':
        from . import qa_em_and_format
        res = qa_em_and_format.compute_score_format(solution_str)
    elif data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score_format(solution_str)
    else:
        raise NotImplementedError
    
    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
    
def _default_compute_score_answer(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == 'hotpotqa/hotpot_qa':
        from . import qa_em_and_format
        res = qa_em_and_format.compute_score_em(solution_str, ground_truth)
    elif data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score_answer(solution_str, ground_truth)
    else:
        raise NotImplementedError
    
    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
    
def _default_compute_score_format_answer(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == 'hotpotqa/hotpot_qa':
        from . import qa_em_and_format
        res = qa_em_and_format.compute_score_format_answer(solution_str, ground_truth)
    elif data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score_format_answer(solution_str, ground_truth)
    else:
        raise NotImplementedError
    
    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])

def _default_compute_all_score(data_source, solution_str, ground_truth, env, extra_info=None):
    if data_source == 'reaction_pathway_search':
        from . import reaction_pathway_reward
        score, end_score, answer_score, format_score = reaction_pathway_reward.compute_all_score(solution_str, ground_truth, env)
        return score, end_score, answer_score, format_score
    else:
        raise NotImplementedError
    

def _default_compute_all_score_back(data_source, solution_str, ground_truth, env, extra_info=None):
    if data_source == 'reaction_pathway_search':
        from . import reaction_pathway_reward_back
        score, end_score, answer_score, format_score = reaction_pathway_reward_back.compute_all_score(solution_str, ground_truth, env)
        return score, end_score, answer_score, format_score
    else:
        raise NotImplementedError