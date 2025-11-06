"""
Implementation of the classic exact-match based evaluation
The core function `compute_entry_f1_scores` replicates the function in https://github.com/SlienceLZJ/ASQP/blob/main/eval_utils.py
"""

import re
# from eval_utils import compute_f1_scores


def read_dict_from_py_file(file_path, dict_name):
    with open(file_path, 'r') as f:
        code = f.read()

    namespace = {}
    exec(code, namespace)
    return namespace[dict_name]



def parse_xml_to_tuple(xml_string, task_type):
    """
    Parse XML-like string into tuple based on task type.
    Returns None if required components are missing or empty (like ABSA evaluator).
    """
    # Extract components
    asp_match = re.search(r'<asp>(.*?)</asp>', xml_string)
    aspect = asp_match.group(1).strip() if asp_match else None

    opn_match = re.search(r'<opn>(.*?)</opn>', xml_string)
    opinion = opn_match.group(1).strip() if opn_match else None

    cat_match = re.search(r'<cat>(.*?)</cat>', xml_string)
    category = cat_match.group(1).strip() if cat_match else None

    sen_match = re.search(r'<sen>(.*?)</sen>', xml_string)
    sentiment = sen_match.group(1).strip() if sen_match else None

    # Validate required components based on task type (like ABSA evaluator)
    if task_type == 'OE':
        if not opinion:  # Opinion is required for all tasks
            return None
        return (opinion,)
    elif task_type == 'AOPE':
        if not opinion or not aspect:  # Both aspect and opinion required
            return None
        return (aspect, opinion)
    elif task_type == 'AOC':
        if not opinion or not aspect or not category:  # All three required
            return None
        return (aspect, opinion, category)
    elif task_type == 'ASTE':
        if not opinion or not aspect or not sentiment:  # All three required
            return None
        return (aspect, opinion, sentiment)
    elif task_type == 'ASQE':
        if not opinion or not aspect or not category or not sentiment:  # All four required
            return None
        return (aspect, opinion, category, sentiment)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def convert_to_f1_format(data_dict):
    """Convert XML-like ASQE format to tuples for backward compatibility"""
    gold_pt = []  # List of lists
    pred_pt = []  # List of lists

    for sample_id, sample_data in data_dict.items():
        task_type = sample_data['task_type']  # Fixed: use sample_data not data_dict[sample_id]

        gold_tuple_list = [parse_xml_to_tuple(raw_str, task_type) for raw_str in sample_data['label']]
        pred_tuple_list = [parse_xml_to_tuple(raw_str, task_type) for raw_str in sample_data['pred']]

        # Filter out None values (invalid items)
        gold_tuple_list = [t for t in gold_tuple_list if t is not None]
        pred_tuple_list = [t for t in pred_tuple_list if t is not None]

        # APPEND each sample's tuples to the overall lists
        gold_pt.append(gold_tuple_list)
        pred_pt.append(pred_tuple_list)

    return pred_pt, gold_pt



def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    # Calculate FP and FN
    fp = n_pred - n_tp  # Predictions not in gold
    fn = n_gold - n_tp  # Gold items not predicted

    print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'n_gold': n_gold, 'n_pred': n_pred, 'tp': n_tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}

    return scores


##=======================================
# usage example

if __name__ == '__main__':

    pretrained_eval_input = {0: {'task_type': 'ASQE',
                                 'input_text': 'Studied for the exam literally the day of... ended with a 90% lol. Not as interesting as I thought it would be tho. The textbook is dry, and the lectures put me to sleep. Overall a good mark booster, but not particularly useful.',
                                 'label': [
                                     '<asp>null</asp><opn>Not as interesting as I thought it would be tho</opn><cat>Course - Overall</cat><sen>negative</sen>',
                                     '<asp>textbook</asp><opn>dry</opn><cat>Course - Course materials</cat><sen>negative</sen>',
                                     '<asp>lectures</asp><opn>put me to sleep</opn><cat>Course - Learning activity</cat><sen>negative</sen>',
                                     '<asp>null</asp><opn>good mark booster</opn><cat>Course - Difficulty</cat><sen>positive</sen>',
                                     '<asp>null</asp><opn>not particularly useful</opn><cat>Course - Overall</cat><sen>negative</sen>'],
                                 'pred': [
                                     '<asp>exam</asp><opn>literally the day of... ended with a 90% lol</opn><cat>Course - Assessment</cat><sen>positive</sen>',
                                     '<asp>null</asp><opn>Not as interesting as I thought it would be tho</opn><cat>Course - Content</cat><sen>negative</sen>',
                                     '<asp>textbook</asp><opn>is dry</opn><cat>Course - Course materials</cat><sen>negative</sen>',
                                     '<asp>lectures</asp><opn>put me to sleep</opn><cat>Course - Learning activity</cat><sen>negative</sen>',
                                     '<asp>null</asp><opn>Overall a good mark booster</opn><cat>Course - Overall</cat><sen>positive</sen>',
                                     '<asp>null</asp><opn>but not particularly useful</opn><cat>Course - Overall</cat><sen>negative</sen>']},
                             1: {'task_type': 'ASQE',
                                 'input_text': 'Studied for the exam literally the day of... ended with a 90% lol. Not as interesting as I thought it would be tho. The textbook is dry, and the lectures put me to sleep. Overall a good mark booster, but not particularly useful.',
                                 'label': [
                                     '<asp>null</asp><opn>Not as interesting as I thought it would be tho</opn><cat>Course - Overall</cat><sen>negative</sen>',
                                     '<asp>textbook</asp><opn>dry</opn><cat>Course - Course materials</cat><sen>negative</sen>',
                                     '<asp>lectures</asp><opn>put me to sleep</opn><cat>Course - Learning activity</cat><sen>negative</sen>',
                                     '<asp>null</asp><opn>good mark booster</opn><cat>Course - Difficulty</cat><sen>positive</sen>',
                                     '<asp>null</asp><opn>not particularly useful</opn><cat>Course - Overall</cat><sen>negative</sen>'],
                                 'pred': [
                                     '<asp>exam</asp><opn>literally the day of... ended with a 90% lol</opn><cat>Course - Assessment</cat><sen>positive</sen>',
                                     '<asp>null</asp><opn>Not as interesting as I thought it would be tho</opn><cat>Course - Content</cat><sen>negative</sen>',
                                     '<asp>textbook</asp><opn>is dry</opn><cat>Course - Course materials</cat><sen>negative</sen>',
                                     '<asp>lectures</asp><opn>put me to sleep</opn><cat>Course - Learning activity</cat><sen>negative</sen>',
                                     '<asp>null</asp><opn>Overall a good mark booster</opn><cat>Course - Overall</cat><sen>positive</sen>',
                                     '<asp>null</asp><opn>but not particularly useful</opn><cat>Course - Overall</cat><sen>negative</sen>']}}

    ## Example of applying exact-match eval method
    pred_tuple_lists, gold_tuples_lists = convert_to_f1_format(pretrained_eval_input)

    scores = compute_f1_scores(pred_tuple_lists, gold_tuples_lists)

    print(f"\033[36mlabels:\033[0m {gold_tuples_lists}\n")
    print(f"\033[36mpreds:\033[0m {pred_tuple_lists}\n")
    print(f"\033[32m{scores}\033[0m")

