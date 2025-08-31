import re
# from math_verify import parse, verify
import json

def remove_duplicates(bbox_list):
    seen = set()
    unique_bboxes = []
    
    for bbox in bbox_list:
        # Convert the position tuple to a tuple for set hashing
        position_tuple = tuple(bbox['Position'])
        
        if position_tuple not in seen:
            seen.add(position_tuple)
            unique_bboxes.append(bbox)
    
    return unique_bboxes

def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection_area = (xi2 - xi1) * (yi2 - yi1)
    
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = area1 + area2 - intersection_area
    
    iou = intersection_area / union_area
    return iou

def compute_reward_iou_v2(iou_results, len_gt):
    iou_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        temp_iou_reward = temp_iou
        if temp_iou == 0:
            temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward
        
    if len_gt>=len(iou_results):
        iou_reward = iou_reward/len_gt
    else:
        iou_reward = iou_reward/len(iou_results)
    return iou_reward

def compute_reward_confidence(iou_results):
    iou_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        temp_iou_reward = temp_iou
        if temp_iou == 0:
            temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward
        
    iou_reward = iou_reward/len(iou_results)
    confidence_reward = confidence_reward/len(iou_results)
    return confidence_reward

def sort_and_calculate_iou(list1, list2, iou_threshold=0.5):
    list2_sorted = sorted(list2, key=lambda x: x['Confidence'], reverse=True)
    
    iou_results = []
    
    matched_list1_indices = set()

    for bbox2 in list2_sorted:
        max_iou = 0
        matched_bbox1 = -1
        best_iou = 0
        for i, bbox1 in enumerate(list1):
            if i not in matched_list1_indices:
                iou = calculate_iou(bbox1['Position'], bbox2['Position'])
                if iou > best_iou:
                    best_iou = iou
                    matched_bbox1 = i

        if best_iou > iou_threshold:
            iou_results.append((best_iou, bbox2['Confidence']))
            matched_list1_indices.add(matched_bbox1)
        else:
            iou_results.append((0, bbox2['Confidence']))
    
    ### [(0.7192676547515258, 1.0), (0, 0.7)]
    return iou_results

def extract_bbox(response):
    start_tag = "<answer>"
    end_tag = "</answer>"
    input_str = response
    # Check if the start tag is in the string
    if start_tag in input_str:
        # Extract the content between the start tag and end tag
        start_idx = input_str.find(start_tag) + len(start_tag)
        end_idx = input_str.find(end_tag)
        
        # If end_tag is not found (i.e., the string is truncated), assume it should be at the end
        if end_idx == -1:
            end_idx = len(input_str)
    
        content_str = input_str[start_idx:end_idx]
    
        # Check if it ends with a closing bracket, if not, fix it
        if not content_str.endswith("]"):
            # If the string is truncated, remove the incomplete part
            content_str = content_str.rsplit("},", 1)[0] + "}]"
    
        # Replace single quotes with double quotes for valid JSON
        content_str_corrected = content_str.replace("'", '"')
    
        # Convert the corrected string to a list of dictionaries (JSON format)
        try:
            bbox_list = json.loads(content_str_corrected)
        except json.JSONDecodeError as e:
            bbox_list = None
    else:
        bbox_list = None
    return bbox_list

def accuracy_reward_iou(predict_str, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""

    reward = 0.0
    # Try symbolic verification first
    # try:
    #     answer = parse(content)
    #     if float(verify(answer, parse(sol))) > 0:
    #         reward = 1.0
    # except Exception:
    #     pass  # Continue to next verification method if this fails

    # student_answer_bbox = []
    # ground_truth_bbox = []
    # iou_results = []
    # show_flage = 0

    # If symbolic verification failed, try string matching
    # if reward == 0.0:
    try:
        show_flage = 1
        # Extract answer from solution if it has think/answer tags
        ground_truth = solution.strip()
        # Extract answer from content if it has think/answer tags
        content_match = re.search(r'<answer>(.*?)</answer>', predict_str)
        student_answer = content_match.group(1).strip() if content_match else predict_str.strip()
        student_answer = '<answer>'+student_answer+'</answer>'

        # fix format error
        student_answer = student_answer.replace("[[",'[')  
        student_answer = student_answer.replace("]]",']')  
        student_answer = student_answer.replace("\n",'')  
        # [{'Position': [254, 303, 291, 365], 'Confidence': 0.9}, {'Position': [100, 100, 200, 200], 'Confidence': 0.8}]
        ground_truth_bbox = extract_bbox(ground_truth)
        student_answer_bbox = extract_bbox(student_answer)
        # pdb.set_trace()
        if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:
            reward = 0.0
        else:
            student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates
            iou_results = sort_and_calculate_iou(ground_truth_bbox, student_answer_bbox)
            ### new iou reward
            reward = compute_reward_iou_v2(iou_results, len(ground_truth_bbox))
            if reward>1:
                reward = 1.0
    except Exception:
        pass  # Keep reward as 0.0 if both methods fail
        
        # import pdb; pdb.set_trace()
        # if os.getenv("DEBUG_MODE") == "true":
        #     log_path = os.getenv("LOG_PATH")
        #     # local_rank = int(os.getenv("LOCAL_RANK", 0))
        #     with open(log_path, "a") as f:
        #         f.write(f"------------- {current_time} Accuracy reward of IoU: {reward} -------------\n")
        #         f.write(f"content: {content}\n")
        #         f.write(f"sol: {sol}\n")
        #         if show_flage==1:
        #             f.write(f"student_answer_bbox: {student_answer_bbox}\n")
        #             f.write(f"ground_truth_bbox: {ground_truth_bbox}\n")
        #             if student_answer_bbox!=None:
        #                 f.write(f"iou_results: {iou_results}\n")
        show_flage = 0 
    return reward

def accuracy_reward_confidence(predict_str: str, solution: str) -> float:
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    # current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    reward = 0.0
    # Try symbolic verification first
    # try:
    #     answer = parse(predict_str)
    #     if float(verify(answer, parse(sol))) > 0:
    #         reward = 1.0
    # except Exception:
    #     pass  # Continue to next verification method if this fails

    student_answer_bbox = []
    ground_truth_bbox = []
    iou_results = []
    show_flage = 0

    # If symbolic verification failed, try string matching
    # if reward == 0.0:
    try:
        show_flage = 1
        # Extract answer from solution if it has think/answer tags
        ground_truth = solution.strip()
        # Extract answer from content if it has think/answer tags
        content_match = re.search(r'<answer>(.*?)</answer>', predict_str)
        student_answer = content_match.group(1).strip() if content_match else predict_str.strip()
        student_answer = '<answer>'+student_answer+'</answer>'

        # fix format error
        student_answer = student_answer.replace("[[",'[')
        student_answer = student_answer.replace("]]",']')
        student_answer = student_answer.replace("\n",'')
        # [{'Position': [254, 303, 291, 365], 'Confidence': 0.9}, {'Position': [100, 100, 200, 200], 'Confidence': 0.8}]
        ground_truth_bbox = extract_bbox(ground_truth)
        student_answer_bbox = extract_bbox(student_answer)
        # pdb.set_trace()
        if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:  # wrong bbox
            reward = 0.0
        else:
            student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates
            iou_results = sort_and_calculate_iou(ground_truth_bbox, student_answer_bbox)
            reward = compute_reward_confidence(iou_results)
            if reward>1:
                reward = 1.0
            if reward<0:
                reward = 0.0
    except Exception:
        pass  # Keep reward as 0.0 if both methods fail
                
        # import pdb; pdb.set_trace()
        # if os.getenv("DEBUG_MODE") == "true":
        #     log_path = os.getenv("LOG_PATH")
        #     # local_rank = int(os.getenv("LOCAL_RANK", 0))
        #     with open(log_path, "a") as f:
        #         f.write(f"------------- {current_time} Accuracy reward of Confidence: {reward} -------------\n")
        #         f.write(f"content: {content}\n")
        #         f.write(f"sol: {sol}\n")
        #         if show_flage==1:
        #             f.write(f"student_answer_bbox: {student_answer_bbox}\n")
        #             f.write(f"ground_truth_bbox: {ground_truth_bbox}\n")
        #             if student_answer_bbox!=None:
        #                 f.write(f"iou_results: {iou_results}\n")
        show_flage = 0 
    return reward


def format_reward(predict_str):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    # pattern = r"<answer>.*?</answer>"
    # matches = [re.match(pattern, content) for content in completion_contents]
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    return 1.0 if match else 0.0

def compute_score(predict_str: str, ground_truth: str) -> float:
    # in predict_str, find the text in the last <|im_start|> <|im_end|>
    # Extract text from the last <|im_start|> <|im_end|> pair
    # print(f"raw predict string: {predict_str}")
    # last_start = predict_str.rfind("<|im_start|>assistant")
    # last_end = predict_str.rfind("<|im_end|>")
    
    # if last_start != -1 and last_end != -1 and last_start < last_end:
    #     predict_str = predict_str[last_start + len("<|im_start|>assistant"):last_end].strip()
    # # If no tags found, use the original string

    predict_str = predict_str.replace("<|im_end|>", "").strip()


    format_reward_num = format_reward(predict_str)
    confidence_reward = accuracy_reward_confidence(predict_str, ground_truth)
    iou_reward = accuracy_reward_iou(predict_str, ground_truth)

    total_reward = format_reward_num + confidence_reward + iou_reward
    # print(f"format_reward_num: {format_reward_num}, confidence_reward: {confidence_reward}, iou_reward: {iou_reward} \n "
    #       f"{predict_str=} \n {ground_truth=}",
    #       flush=True)

    return total_reward