import json

def filter_dataset(input_file, output_file, rv_condition, cd_condition):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filtered_data = []
    
    for item in data:
        if item.get('logical_correctness', 0) == 1 and \
           rv_condition(item.get('reasoning_verbosity', 0)) and \
           cd_condition(item.get('cognitive_difficulty', 0)):
            
            filtered_item = {
                'instruction': item['instruction'],
                'output': item['output']
            }
            filtered_data.append(filtered_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print(f"筛选完成，共找到{len(filtered_data)}条符合条件的记录，已保存到{output_file}")

if __name__ == "__main__":
    def rv_condition(score):
        return score >= 3 and score <=5
    
    def cd_condition(score):
        return score == 4
    
    filter_dataset(
        input_file='input.json',  
        output_file='filtered_output.json',  
        rv_condition=rv_condition,
        cd_condition=cd_condition
    )