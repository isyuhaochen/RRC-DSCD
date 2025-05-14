import re
import ast

def accuracy_reward(completions, **kwargs):
    labels = kwargs.get("label", None)
    evidences = kwargs.get("evidence", None)
    """Reward function that checks if the completion is the same as the ground truth."""
    rewards = []
    for content, label, evidences in zip(completions, labels, evidences):
        try:
            # Extract the answer from the completion
            text = content.split("<answer>")[1].split("</answer>")[0].lower()
            beginning = text.find("\njudgment") + len("\njudgment")
            ending = text.find("\n", beginning)
            judge = text[beginning: ending]
            if 'yes' in judge and label:
                reward = 1.0
            elif 'no' in judge and not label:
                reward = 1.0
            else:
                reward = 0.0

            if label:
                evidence_str = text.split("evidence:")[1].strip().replace("\n", "")
                match = re.search(r'\[.*\]', evidence_str)  # 匹配方括号内的内容
                parsed_list = ast.literal_eval(match.group(0))
                
                sign = 1
                for evidence in parsed_list:
                    for label_evidence in evidences:
                        if evidence.lower() in label_evidence.lower() or label_evidence.lower() in evidence.lower():
                            reward += 1/len(parsed_list)
                            sign = 0
                            break
                if sign:
                    reward -= 0.5
            
        except Exception as e:
            # If the completion is not parseable, we reward 0 to skip this example
            print("Failed to parse completion: ", content)
            rewards.append(0.0)
            continue
       
        rewards.append(reward)
        # print('accuracy_reward:', rewards)
    return rewards

def format_reward(completions, **kwargs):
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    # print('format_reward:', [1.0 if match else 0.0 for match in matches])
    return [1.0 if match else 0.0 for match in matches]
 
def sentence_cover_reward(completions, **kwargs):
    doc_sentnece_nums = kwargs.get("doc_sentnece_num", 0)
    rewards = []

    def parse_ranges(range_str):
        numbers = set()
        matches = re.findall(r'\[(\d+)(?:-(\d+))?\]', range_str)

        for match in matches:
            start = int(match[0])
            end = int(match[1]) if match[1] else start  
            numbers.update(range(start, end + 1))
        return numbers
    
    for content,doc_sentnece_num in zip(completions,doc_sentnece_nums):
        try:
            reasoning_content = content.split("<think>")[1].split("</think>")[0].lower()
            length = len(parse_ranges(reasoning_content))
            reward = length / doc_sentnece_num 
        except Exception as e:
            rewards.append(0.0)
            continue
        rewards.append(reward)
    return rewards

def answer_sentence_cover_reward(completions, **kwargs):
    evidence_label_lists = kwargs.get("evidence_label_list", [])
    labels = kwargs.get("label", None)
    rewards = []
    def parse_ranges(range_str):
        numbers = set()
        matches = re.findall(r'\[(\d+)(?:-(\d+))?\]', range_str)

        for match in matches:
            start = int(match[0])
            end = int(match[1]) if match[1] else start  
            numbers.update(range(start, end + 1))
        return numbers
    for content,evidence_label_list,label in zip(completions, evidence_label_lists,labels):
        if not label:
            rewards.append(1.0)
            continue
        try:
            reward = 0.0
            reasoning_content = content.split("<think>")[1].split("</think>")[0].lower()
            numbers = parse_ranges(reasoning_content)
            for label in evidence_label_list:
                if label in numbers:
                    reward = 1.0
        except Exception as e:
            rewards.append(0.0)
            continue
        rewards.append(reward)
    # print('answer_sentence_cover_reward:', rewards)
    return rewards

def tag_count_reward(completions, **kwargs):
    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count
    # print('tag_count_reward:', [count_tags(c) for c in completions])
    return [count_tags(c) for c in completions]

def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        rewards = []
        for completion in completions:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        # print(f"repetition_penalty_reward: {rewards}")
        return rewards

    return repetition_penalty_reward

def get_reward_funcs():
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "sentence_cover_reward": sentence_cover_reward,
        "answer_sentence_cover_reward": answer_sentence_cover_reward,
        "tag_count": tag_count_reward,
        "repetition_penalty": get_repetition_penalty_reward(ngram_size=4,max_penalty=-2),

    }
    # reward_funcs_list = ['accuracy', 'format', 'sentence_cover_reward', 'answer_sentence_cover_reward', 'tag_count', 'repetition_penalty']
    # reward_funcs_list = ['accuracy', 'format', 'sentence_cover_reward']
    # reward_funcs_list = ['accuracy', 'format', 'sentence_cover_reward','repetition_penalty']
    # reward_funcs_list = ['accuracy', 'format','repetition_penalty']
    # reward_funcs_list = ['accuracy', 'format']
    # reward_funcs_list = ['answer_sentence_cover_reward']

    # reward_funcs_list = ['accuracy', 'format', 'sentence_cover_reward','answer_sentence_cover_reward']
    reward_funcs_list = ['format']
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in reward_funcs_list]

    return reward_funcs
