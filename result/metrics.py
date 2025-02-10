import json


def evaluate_QA(result_file):
    answer_cnt = [0, 0, 0]
    goal_answer_cnt = [0, 0, 0]
    correct_propotion = [0, 0, 0]
    
    with open(result_file, 'r') as f:
        QA_results = json.load(f)

    correct_cnt = 0
    total_count = 0
    error_parse_cnt = 0
    for sample in QA_results:
        gold_answer = sample['label']
        try:
            if sample['model_answer'].endswith('"answer": "A"\n}') or sample['model_answer'].startswith('{\n  "answer": "A"\n}'):
                sample['model_answer'] = '{\n  "answer": "A"\n}'
            elif sample['model_answer'].endswith('"answer": "B"\n}') or sample['model_answer'].startswith('{\n  "answer": "B"\n}'):
                sample['model_answer'] = '{\n  "answer": "B"\n}'
            elif sample['model_answer'].endswith('"answer": "C"\n}') or sample['model_answer'].startswith('{\n  "answer": "C"\n}'):
                # answer": "C"\n}
                sample['model_answer'] = '{\n  "answer": "C"\n}'
                
            prediction = eval(sample['model_answer'].replace('```json', '').replace('```', '').split("\n\n")[0])['answer']
                
            if prediction == gold_answer:
                correct_cnt += 1
                if prediction == "A":
                    correct_propotion[0] += 1
                elif prediction == "B":
                    correct_propotion[1] += 1
                elif prediction == "C":
                    correct_propotion[2] += 1
            
            print(f"prediction: {prediction} \t gold_answers: {gold_answer} \t match: {prediction == gold_answer}")
            
            if prediction == "A":
                answer_cnt[0] += 1
            elif prediction == "B":
                answer_cnt[1] += 1
            elif prediction == "C":
                answer_cnt[2] += 1
                
            if gold_answer == "A":
                goal_answer_cnt[0] += 1
            elif gold_answer == "B":
                goal_answer_cnt[1] += 1
            elif gold_answer == "C":
                goal_answer_cnt[2] += 1
        except:
            error_parse_cnt += 1

        total_count += 1
    
    avg_acc = correct_cnt / total_count
    print(f"Acc: {avg_acc}")
    print(f"Correct: {correct_cnt}. Total: {total_count}")
    print(f"Parse Error: {error_parse_cnt}")
    
    print(f"Goal Answer: {goal_answer_cnt}")
    print(f"Model Answer: {answer_cnt}")
    print(f"Correct: {correct_propotion}")


def evaluate_FOLIO(result_file):
    # load filtered version
    with open('logic_data/FOLIO/filtered_version.json', 'r') as f:
        filtered_folio = json.load(f)

    filtered_dic = {}
    for item in filtered_folio:
        filtered_dic[item['id']] = item['answer']
    
    answer_cnt = [0, 0, 0]
    goal_answer_cnt = [0, 0, 0]
    correct_propotion = [0, 0, 0]
    
    with open(result_file, 'r') as f:
        QA_results = json.load(f)

    correct_cnt = 0
    total_count = 0
    error_parse_cnt = 0
    for sample in QA_results:
        if sample['id'] not in filtered_dic.keys():
            continue
        
        gold_answer = filtered_dic[sample['id']]
        try:
            prediction = eval(sample['model_answer'].replace('```json', '').replace('```', '').split("\n\n")[0])['answer']
                
            if prediction == gold_answer:
                correct_cnt += 1
                if prediction == "A":
                    correct_propotion[0] += 1
                elif prediction == "B":
                    correct_propotion[1] += 1
                elif prediction == "C":
                    correct_propotion[2] += 1
            
            print(f"prediction: {prediction} \t gold_answers: {gold_answer} \t match: {prediction == gold_answer}")
            
            if prediction == "A":
                answer_cnt[0] += 1
            elif prediction == "B":
                answer_cnt[1] += 1
            elif prediction == "C":
                answer_cnt[2] += 1
                
            if gold_answer == "A":
                goal_answer_cnt[0] += 1
            elif gold_answer == "B":
                goal_answer_cnt[1] += 1
            elif gold_answer == "C":
                goal_answer_cnt[2] += 1
        except:
            error_parse_cnt += 1

        total_count += 1
    
    avg_acc = correct_cnt / total_count
    print(f"Acc: {avg_acc}")
    print(f"Correct: {correct_cnt}. Total: {total_count}")
    print(f"Parse Error: {error_parse_cnt}")
    
    print(f"Goal Answer: {goal_answer_cnt}")
    print(f"Model Answer: {answer_cnt}")
    print(f"Correct: {correct_propotion}")

def evaluate_claude_cot(result_file):
    answer_cnt = [0, 0, 0]
    goal_answer_cnt = [0, 0, 0]
    correct_propotion = [0, 0, 0]
    
    with open(result_file, 'r') as f:
        QA_results = json.load(f)

    correct_cnt = 0
    total_count = 0
    error_parse_cnt = 0
    for sample in QA_results:
        gold_answer = sample['label']
        
        try:
            if sample['model_answer'].endswith("\n  \"answer\": \"A\"\n}"):
                prediction = "A"
            elif sample['model_answer'].endswith("\n  \"answer\": \"B\"\n}"):
                prediction = "B"
            elif sample['model_answer'].endswith("\n  \"answer\": \"C\"\n}"):
                prediction = "C"
            else:
                assert 1 == 2
                
            if prediction == gold_answer:
                correct_cnt += 1
                if prediction == "A":
                    correct_propotion[0] += 1
                elif prediction == "B":
                    correct_propotion[1] += 1
                elif prediction == "C":
                    correct_propotion[2] += 1
            
            print(f"prediction: {prediction} \t gold_answers: {gold_answer} \t match: {prediction == gold_answer}")
            
            if prediction == "A":
                answer_cnt[0] += 1
            elif prediction == "B":
                answer_cnt[1] += 1
            elif prediction == "C":
                answer_cnt[2] += 1
                
            if gold_answer == "A":
                goal_answer_cnt[0] += 1
            elif gold_answer == "B":
                goal_answer_cnt[1] += 1
            elif gold_answer == "C":
                goal_answer_cnt[2] += 1
        except:
            error_parse_cnt += 1

        total_count += 1
    
    avg_acc = correct_cnt / total_count
    print(f"Acc: {avg_acc}")
    print(f"Correct: {correct_cnt}. Total: {total_count}")
    print(f"Parse Error: {error_parse_cnt}")
    
    print(f"Goal Answer: {goal_answer_cnt}")
    print(f"Model Answer: {answer_cnt}")
    print(f"Correct: {correct_propotion}")

def evaluate_claude_cot_folio(result_file):
    # load filtered version
    with open('logic_data/FOLIO/filtered_version.json', 'r') as f:
        filtered_folio = json.load(f)

    filtered_dic = {}
    for item in filtered_folio:
        filtered_dic[item['id']] = item['answer']
        
    answer_cnt = [0, 0, 0]
    goal_answer_cnt = [0, 0, 0]
    correct_propotion = [0, 0, 0]
    
    with open(result_file, 'r') as f:
        QA_results = json.load(f)

    correct_cnt = 0
    total_count = 0
    error_parse_cnt = 0
    for sample in QA_results:
        if sample['id'] not in filtered_dic.keys():
            continue
        
        gold_answer = sample['label']
        try:
            if sample['model_answer'].endswith("\n  \"answer\": \"A\"\n}"):
                prediction = "A"
            elif sample['model_answer'].endswith("\n  \"answer\": \"B\"\n}"):
                prediction = "B"
            elif sample['model_answer'].endswith("\n  \"answer\": \"C\"\n}"):
                prediction = "C"
            else:
                assert 1 == 2
                
            if prediction == gold_answer:
                correct_cnt += 1
                if prediction == "A":
                    correct_propotion[0] += 1
                elif prediction == "B":
                    correct_propotion[1] += 1
                elif prediction == "C":
                    correct_propotion[2] += 1
            
            print(f"prediction: {prediction} \t gold_answers: {gold_answer} \t match: {prediction == gold_answer}")
            
            if prediction == "A":
                answer_cnt[0] += 1
            elif prediction == "B":
                answer_cnt[1] += 1
            elif prediction == "C":
                answer_cnt[2] += 1
                
            if gold_answer == "A":
                goal_answer_cnt[0] += 1
            elif gold_answer == "B":
                goal_answer_cnt[1] += 1
            elif gold_answer == "C":
                goal_answer_cnt[2] += 1
        except:
            error_parse_cnt += 1

        total_count += 1
    
    avg_acc = correct_cnt / total_count
    print(f"Acc: {avg_acc}")
    print(f"Correct: {correct_cnt}. Total: {total_count}")
    print(f"Parse Error: {error_parse_cnt}")
    
    print(f"Goal Answer: {goal_answer_cnt}")
    print(f"Model Answer: {answer_cnt}")
    print(f"Correct: {correct_propotion}")


def evaluate_claude_direct(result_file):
    answer_cnt = [0, 0, 0]
    goal_answer_cnt = [0, 0, 0]
    correct_propotion = [0, 0, 0]
    
    with open(result_file, 'r') as f:
        QA_results = json.load(f)

    correct_cnt = 0
    total_count = 0
    error_parse_cnt = 0
    for sample in QA_results:
        gold_answer = sample['label']
        try:
            if '{\n  \"answer\": \"A\"\n}' in sample['model_answer']:
                prediction = "A"
            elif '{\n  \"answer\": \"B\"\n}' in sample['model_answer']:
                prediction = "B"
            elif '{\n  \"answer\": \"C\"\n}' in sample['model_answer']:
                prediction = "C"
            else:
                assert 1 == 2
                
            if prediction == gold_answer:
                correct_cnt += 1
                if prediction == "A":
                    correct_propotion[0] += 1
                elif prediction == "B":
                    correct_propotion[1] += 1
                elif prediction == "C":
                    correct_propotion[2] += 1
            
            print(f"prediction: {prediction} \t gold_answers: {gold_answer} \t match: {prediction == gold_answer}")
            
            if prediction == "A":
                answer_cnt[0] += 1
            elif prediction == "B":
                answer_cnt[1] += 1
            elif prediction == "C":
                answer_cnt[2] += 1
                
            if gold_answer == "A":
                goal_answer_cnt[0] += 1
            elif gold_answer == "B":
                goal_answer_cnt[1] += 1
            elif gold_answer == "C":
                goal_answer_cnt[2] += 1
        except:
            error_parse_cnt += 1

        total_count += 1
    
    avg_acc = correct_cnt / total_count
    print(f"Acc: {avg_acc}")
    print(f"Correct: {correct_cnt}. Total: {total_count}")
    print(f"Parse Error: {error_parse_cnt}")
    
    print(f"Goal Answer: {goal_answer_cnt}")
    print(f"Model Answer: {answer_cnt}")
    print(f"Correct: {correct_propotion}")

def evaluate_claude_direct_folio(result_file):
    # load filtered version
    with open('logic_data/FOLIO/filtered_version.json', 'r') as f:
        filtered_folio = json.load(f)

    filtered_dic = {}
    for item in filtered_folio:
        filtered_dic[item['id']] = item['answer']
        
    answer_cnt = [0, 0, 0]
    goal_answer_cnt = [0, 0, 0]
    correct_propotion = [0, 0, 0]
    
    with open(result_file, 'r') as f:
        QA_results = json.load(f)

    correct_cnt = 0
    total_count = 0
    error_parse_cnt = 0
    for sample in QA_results:
        if sample['id'] not in filtered_dic.keys():
            continue
        
        gold_answer = sample['label']
        try:
            if '{\n  \"answer\": \"A\"\n}' in sample['model_answer']:
                prediction = "A"
            elif '{\n  \"answer\": \"B\"\n}' in sample['model_answer']:
                prediction = "B"
            elif '{\n  \"answer\": \"C\"\n}' in sample['model_answer']:
                prediction = "C"
            else:
                assert 1 == 2
                
            if prediction == gold_answer:
                correct_cnt += 1
                if prediction == "A":
                    correct_propotion[0] += 1
                elif prediction == "B":
                    correct_propotion[1] += 1
                elif prediction == "C":
                    correct_propotion[2] += 1
            
            print(f"prediction: {prediction} \t gold_answers: {gold_answer} \t match: {prediction == gold_answer}")
            
            if prediction == "A":
                answer_cnt[0] += 1
            elif prediction == "B":
                answer_cnt[1] += 1
            elif prediction == "C":
                answer_cnt[2] += 1
                
            if gold_answer == "A":
                goal_answer_cnt[0] += 1
            elif gold_answer == "B":
                goal_answer_cnt[1] += 1
            elif gold_answer == "C":
                goal_answer_cnt[2] += 1
        except:
            error_parse_cnt += 1

        total_count += 1
    
    avg_acc = correct_cnt / total_count
    print(f"Acc: {avg_acc}")
    print(f"Correct: {correct_cnt}. Total: {total_count}")
    print(f"Parse Error: {error_parse_cnt}")
    
    print(f"Goal Answer: {goal_answer_cnt}")
    print(f"Model Answer: {answer_cnt}")
    print(f"Correct: {correct_propotion}")
    
# {\n  \"answer\": \"C\"\n}
def evaluate_mistral(result_file):
    answer_cnt = [0, 0, 0]
    goal_answer_cnt = [0, 0, 0]
    correct_propotion = [0, 0, 0]
    
    with open(result_file, 'r') as f:
        QA_results = json.load(f)

    correct_cnt = 0
    total_count = 0
    error_parse_cnt = 0
    for sample in QA_results:
        gold_answer = sample['label']
        try:
            sample['model_answer'] = sample['model_answer'].split("\n\n")[0]
            sample['model_answer'] = sample['model_answer'].split("\n------\n")[0]
            if '\n  "answer": "A"\n' in sample['model_answer'] or "A) True" in sample['model_answer']:
                prediction = "A"
            elif '\n  "answer": "B"\n' in sample['model_answer'] or "B) False" in sample['model_answer']:
                prediction = "B"
            elif '\n  "answer": "C"\n' in sample['model_answer'] or "C) Uncertain" in sample['model_answer']:
                prediction = "C"
            else:
                raise ValueError("Error")
                
            if prediction == gold_answer:
                correct_cnt += 1
                if prediction == "A":
                    correct_propotion[0] += 1
                elif prediction == "B":
                    correct_propotion[1] += 1
                elif prediction == "C":
                    correct_propotion[2] += 1
            
            print(f"prediction: {prediction} \t gold_answers: {gold_answer} \t match: {prediction == gold_answer}")
            
            if prediction == "A":
                answer_cnt[0] += 1
            elif prediction == "B":
                answer_cnt[1] += 1
            elif prediction == "C":
                answer_cnt[2] += 1
                
            if gold_answer == "A":
                goal_answer_cnt[0] += 1
            elif gold_answer == "B":
                goal_answer_cnt[1] += 1
            elif gold_answer == "C":
                goal_answer_cnt[2] += 1
        except:
            error_parse_cnt += 1

        total_count += 1
    
    avg_acc = correct_cnt / total_count
    print(f"Acc: {avg_acc}")
    print(f"Correct: {correct_cnt}. Total: {total_count}")
    print(f"Parse Error: {error_parse_cnt}")
    
    print(f"Goal Answer: {goal_answer_cnt}")
    print(f"Model Answer: {answer_cnt}")
    print(f"Correct: {correct_propotion}")
    

def evaluate_mistral_folio(result_file):
    # load filtered version
    with open('logic_data/FOLIO/filtered_version.json', 'r') as f:
        filtered_folio = json.load(f)

    filtered_dic = {}
    for item in filtered_folio:
        filtered_dic[item['id']] = item['answer']
        
        
    answer_cnt = [0, 0, 0]
    goal_answer_cnt = [0, 0, 0]
    correct_propotion = [0, 0, 0]
    
    with open(result_file, 'r') as f:
        QA_results = json.load(f)

    correct_cnt = 0
    total_count = 0
    error_parse_cnt = 0
    for sample in QA_results:
        if sample['id'] not in filtered_dic.keys():
            continue
        
        gold_answer = sample['label']
        try:
            sample['model_answer'] = sample['model_answer'].split("\n------\n")[0]
            if '\n  "answer": "A"\n}' in sample['model_answer']:
                prediction = "A"
            elif '\n  "answer": "B"\n}' in sample['model_answer']:
                prediction = "B"
            elif '\n  "answer": "C"\n}' in sample['model_answer']:
                prediction = "C"
            else:
                raise ValueError("Error")
                
            if prediction == gold_answer:
                correct_cnt += 1
                if prediction == "A":
                    correct_propotion[0] += 1
                elif prediction == "B":
                    correct_propotion[1] += 1
                elif prediction == "C":
                    correct_propotion[2] += 1
            
            print(f"prediction: {prediction} \t gold_answers: {gold_answer} \t match: {prediction == gold_answer}")
            
            if prediction == "A":
                answer_cnt[0] += 1
            elif prediction == "B":
                answer_cnt[1] += 1
            elif prediction == "C":
                answer_cnt[2] += 1
                
            if gold_answer == "A":
                goal_answer_cnt[0] += 1
            elif gold_answer == "B":
                goal_answer_cnt[1] += 1
            elif gold_answer == "C":
                goal_answer_cnt[2] += 1
        except:
            error_parse_cnt += 1

        total_count += 1
    
    avg_acc = correct_cnt / total_count
    print(f"Acc: {avg_acc}")
    print(f"Correct: {correct_cnt}. Total: {total_count}")
    print(f"Parse Error: {error_parse_cnt}")
    
    print(f"Goal Answer: {goal_answer_cnt}")
    print(f"Model Answer: {answer_cnt}")
    print(f"Correct: {correct_propotion}")


if __name__ == "__main__":
    filename = "RESULT_FILE"
    
    if "folio" in filename.lower():
        if "mistral" in filename.lower():
            evaluate_mistral_folio(filename)
        elif "claude" in filename.lower():
            if 'CoT_' in filename:
                evaluate_claude_cot_folio(filename)
            else:
                evaluate_claude_direct_folio(filename)
        else:
            evaluate_FOLIO(filename)
    else:
        if 'mistral' in filename.lower():
            evaluate_mistral(filename)
        elif 'claude' in filename.lower():
            if 'CoT_' in filename:
                evaluate_claude_cot(filename)
            else:
                evaluate_claude_direct(filename)
        else:
            evaluate_QA(filename)
    
    
    
