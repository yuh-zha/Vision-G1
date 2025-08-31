import re
from math_verify import parse, verify, StringExtractionConfig
import json

def format_reward(predict_str: str) -> float:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    # pattern = r"<answer>.*?</answer>"
    # matches = [re.match(pattern, content) for content in completion_contents]
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    return 1.0 if match else 0.0

def _clean_answer_content(answer_content: str) -> str:
    # extract Options from e.g., (A) --> A
    # TODO: add later
    ''' Extract the content inside the () '''
    # cleaned = answer_content.strip()
    # match = re.search(r"\(([A-Za-z])\)", cleaned)
    # if match:
    #     return match.group(1)
    # return cleaned

    cleaned = answer_content.strip()
    # extract the string after Final Answer, case insensitive:
    match = re.search(r'(?i)final answer:?[\s]*(.*)', cleaned)
    if match:
        return match.group(1).strip()
    return cleaned

def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    inboxed_string = string[left_brace_idx + 1:right_brace_idx].strip()
    # if \text{} exists, extract content inside \text{}
    match = re.search(r"\\text\{(.*?)\}", inboxed_string)
    if match:
        return match.group(1).strip()
    # if \text{} not exists, return the whole string
    return inboxed_string

def acc_reward(predict_str: str, ground_truth: str) -> float:
    # first use math_verify, if fails use string match
    if verify(parse(predict_str), parse(ground_truth)):
        return 1.0

    # match string in <answer> </answer>
    answers = list(re.finditer(r"<answer>(.*?)</answer>", predict_str, re.DOTALL))
    if answers:
        last_match = answers[-1]
        answer_content = _clean_answer_content(last_match.group(1).strip())
        if verify(parse("$"+answer_content+"$"), parse("$"+ground_truth+"$")):
            return 1.0
        if answer_content.lower() == ground_truth.strip().lower():
            return 1.0

    # match string in \boxed, usually it's math
    answer_content = _last_boxed_only_string(predict_str)
    if answer_content:
        if verify(parse("$"+answer_content+"$"), parse("$"+ground_truth+"$")):
            return 1.0
        if answer_content.lower() == ground_truth.strip().lower():
            return 1.0
    
          
    
    return 0.0

def compute_score(predict_str: str, ground_truth: str) -> dict:
    # remove <|im_end|> first
    predict_str = predict_str.replace("<|im_end|>", "").strip()


    # format_reward_num = format_reward(predict_str)
    # acc_reward_num = acc_reward(predict_str, ground_truth)

    # total_reward = format_reward_num + acc_reward_num
    # print(f"format_reward_num: {format_reward_num}, confidence_reward: {confidence_reward}, iou_reward: {iou_reward} \n "
    #       f"{predict_str=} \n {ground_truth=}",
    #       flush=True)

    total_reward = acc_reward(predict_str, ground_truth)
    if total_reward == 0.0:
        pass
        # print(f"[INFO: 0 Acc Reward] predict_str: {predict_str}, ground_truth: {ground_truth}") 
    return {
        "score": total_reward,
        "acc": total_reward,
    }

def compute_score_boxed(predict_str: str, ground_truth: str) -> dict:
    final_reward = 0.0
    predict_str = predict_str.replace("<|im_end|>", "").strip()

    if verify(parse(predict_str), parse(ground_truth)):
        final_reward = 1.0

    answer_content = _last_boxed_only_string(predict_str)

    if answer_content:
        if verify(parse("$"+answer_content+"$"), parse("$"+ground_truth+"$")):
            final_reward = 1.0
        if answer_content.lower() == ground_truth.strip().lower():
            final_reward = 1.0
    
    return {
        "score": final_reward,
        "acc": final_reward,
    }

def compute_score_boxed_iou(predict_str: str, ground_truth: str) -> dict:
    def _iou(boxA, boxB):
        """
        Compute the Intersection over Union (IoU) of two boxes.
        
        Parameters
        ----------
        boxA : list or tuple of 4 numbers [x1, y1, x2, y2]
        boxB : list or tuple of 4 numbers [x1, y1, x2, y2]
        
        Returns
        -------
        float
            IoU value in [0, 1]
        """
        # Unpack coordinates
        xA1, yA1, xA2, yA2 = boxA
        xB1, yB1, xB2, yB2 = boxB

        # Compute the (x, y)-coordinates of the intersection rectangle
        xI1 = max(xA1, xB1)
        yI1 = max(yA1, yB1)
        xI2 = min(xA2, xB2)
        yI2 = min(yA2, yB2)

        # Compute width and height of the intersection rectangle
        inter_w = max(0, xI2 - xI1)
        inter_h = max(0, yI2 - yI1)
        inter_area = inter_w * inter_h

        # Compute areas of the input boxes
        areaA = max(0, xA2 - xA1) * max(0, yA2 - yA1)
        areaB = max(0, xB2 - xB1) * max(0, yB2 - yB1)

        # Compute union
        union_area = areaA + areaB - inter_area

        # Avoid division by zero
        if union_area == 0:
            return 0.0

        return inter_area / union_area

    final_reward = 0.0
    answer_content = _last_boxed_only_string(predict_str)
    if answer_content is None:
        return {
            "score": 0.0,
            "acc": 0.0,
        }
    import re

    pattern = r'\[\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\]'
    preds = re.findall(pattern, answer_content)
    gt = re.findall(pattern, ground_truth)

    if preds and gt:
        preds = preds[0]
        gt = gt[0]
    else:
        return {
            "score": 0.0,
            "acc": 0.0,
        }

    final_reward = _iou(preds, gt)
    return {
        "score": final_reward,
        "acc": final_reward,
    }
    
    
    






