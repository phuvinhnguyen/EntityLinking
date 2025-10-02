from LLM_calls import load_llm, llm_call
import json
from tqdm import tqdm
import random
import argparse

def read_prompt(file_name):
    prompt_dict = {}
    with open(file_name) as prompt_f:
        for line in prompt_f:
            line = json.loads(line.strip())
            prompt_dict[line['id']] = line
    
    return prompt_dict

def list_prompt_formula(src_dict:dict):
    content = 'Mention:{}\n'.format(src_dict['mention'])

    context = src_dict['left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['right_context']
    context = context.strip()
    context = ' '.join(context.split())
    content += 'Context:{}\n'.format(context)

    candidates = random.sample(src_dict['candidates'], len(src_dict['candidates']))
    i = 1
    for cand in candidates:
        cand_entity = '{}.{}'.format(cand['name'], cand['summary'])
        content += 'Entity {}:{}\n'.format(i, cand_entity)
        i += 1
    content += '\n'
    return content

def read_cot(cot_file_name):
    cot_index_dict = {}
    i = 0
    with open(cot_file_name) as input_f:
        for line in tqdm(input_f):
            line = json.loads(line.strip())
            prompt = list_prompt_formula(line)
            prompt += 'Answer: {}\n\n'.format(line['gpt_ans'].replace('\n',''))
            
            cot_index_dict[i] = prompt
            i += 1

    return cot_index_dict


def summary_prompt(src_dict):
    content = 'The following is a description of {}. Please extract the key information of {} and summarize it in one sentence.\n\n{}'.format(
    src_dict['title'], src_dict['title'], src_dict['text'])

    return content

def category_prompt(src_dict):
    category_list = ['General reference','Culture and the arts','Geography and places','Health and fitness', 'History and events', 'Human activities', 'Mathematics and logic', 'Natural and physical sciences', 'People and self', 'Philosophy and thinking', 'Religion and belief systems', 'Society and social sciences', 'Technology and applied sciences']
    content = 'You are a mention classifier. Wikipedia categorizes entity into the following categories:'
    for category in category_list:
        content += ' {},'.format(category)
    content = (content.rstrip(',') + '.\n')
    content += "Now, I will give you a mention and its context, the mention will be highlighted with '###'.\n\n"""

    content += 'mention:{}\n'.format(src_dict['mention'])
    if 'cut_left_context' in src_dict.keys():
        context = src_dict['cut_left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['cut_right_context']
    else:
        context = src_dict['left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['right_context']
    context = context.strip()
    context = ' '.join(context.split())
    content += 'context:{}\n\n'.format(context)

    content += "please determine which of the above categories the mention {} belongs to?".format(src_dict['mention'])
    return content

def point_wise_el_prompt(src_dict, instruction_dict, dataset, cut=False):
    if dataset == 'zeshel':
        content = "You're an entity disambiguator. I'll give you the description of entity disambiguation and some tips on entity disambiguation, and you need to pay attention to these textual features:\n\n"
        content += instruction_dict[0]['prompt']
        content += '\n\n'
        content += "Now, I'll give you a mention, a context, and a candidate entity, and the mention will be highlighted with '###'.\n\n"
        content += 'Mention:{}\n'.format(src_dict['mention'])

        if cut:
            context = src_dict['cut_left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['cut_right_context']
        else:
            context = src_dict['left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['right_context']
        context = context.strip()
        context = ' '.join(context.split())
        content += 'Context:{}\n'.format(context)

        cand_entity = '{}.{}'.format(src_dict['candidates']['title'], src_dict['candidates']['summary'])
        content += 'Candidate Entity:{}\n\n'.format(cand_entity)

        content += """You need to determine if the mention and the candidate entity are related. Please refer to the above tips and give your reasons, and finally answer 'yes' or 'no'. Answer 'yes' when you think the information is insufficient or uncertain."""
    else:
        content = "You're an entity disambiguator. I'll give you the description of entity disambiguation and some tips on entity disambiguation, and you need to pay attention to these textual features:\n\n"
        content += instruction_dict[0]['prompt']
        content += '\n\n'
        content += "Now, I'll give you a mention, a context, and a candidate entity, and the mention will be highlighted with '###'.\n\n"
        content += 'mention:{}\n'.format(src_dict['mention'])

        context = src_dict['left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['right_context']
        context = context.strip()
        context = ' '.join(context.split())
        content += 'context:{}\n'.format(context)

        cand_entity = '{}.{}'.format(src_dict['cand_name'], src_dict['cand_text'])
        content += 'candidate entity:{}\n\n'.format(cand_entity)

        content += """You need to determine if the mention and the candidate entity are related. Please refer to the above tips and give your reasons, and finally answer 'yes' or 'no'. Answer 'yes' when you think the information is insufficient or uncertain."""

    return content

def prior_prompt(src_dict, datasets):
    system_content = "You're an entity disambiguator."
    content = "I'll provide you a mention and its candidates below.\n\n"
    content += 'Mention:{}\n'.format(src_dict['mention'])

    candidates = random.sample(src_dict['candidates'], len(src_dict['candidates']))
    i = 1
    for cand in candidates:
        if datasets == 'zeshel':
            cand_entity = '{}.{}'.format(cand['title'], cand['summary'])
            content += 'Entity {}:{}\n'.format(cand['document_id'], cand_entity)
        else:
            cand_entity = '{}.{}'.format(cand['name'], cand['summary'])
            content += 'Entity {}:{}\n'.format(i, cand_entity)
        i += 1
    content += '\n'

    content += 'Based on your knowledge, please determine which is the most likely entity when people refer to mention "{}", '.format(src_dict['mention'])
    content += "and finally answer the id and name of the entity."

    return system_content, content

def context_prompt(src_dict, datasets, cot_index_dict, instruction_dict, prompt_id=0, ent_des='summary', cut=False):
    system_content = "You're an entity disambiguator. I'll give you the description of entity disambiguation and some tips on entity disambiguation, you should pay attention to these textual features:\n\n"
    system_content += instruction_dict[prompt_id]['prompt']

    '''category and sentence sim'''
    cot_index = src_dict['cot_index']
    cot_case = cot_index_dict[cot_index]
    
    content = 'The following example will help you understand the task:\n\n'
    content += cot_case

    content += "Now, I'll give you a mention, a context, and a list of candidates entities, the mention will be highlighted with '###' in context.\n\n"
    content += 'Mention:{}\n'.format(src_dict['mention'])

    if ('cut_left_context' in src_dict.keys()) and cut:
        context = src_dict['cut_left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['cut_right_context']
    else:
        context = src_dict['left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['right_context']
    context = context.strip()
    context = ' '.join(context.split())
    content += 'Context:{}\n'.format(context)

    candidates = random.sample(src_dict['candidates'], len(src_dict['candidates']))
    i = 1
    for cand in candidates:
        if datasets == 'zeshel':
            cand_entity = '{}.{}'.format(cand['title'], cand[ent_des])
            content += 'Entity {}:{}\n'.format(cand['document_id'], cand_entity)
        else:
            cand_entity = '{}.{}'.format(cand['name'], cand[ent_des])
            content += 'Entity {}:{}\n'.format(i, cand_entity)
        i += 1
    content += '\n'

    content += """You need to determine which candidate entity is more likely to be the mention. Please refer to the above example, give your reasons, and finally answer id of the entity and the name of the entity. If all candidate entities are not appropriate, you can answer '-1.None'."""

    return system_content, content

def merge_prompt(src_dict, datasets, instruction_dict, prompt_id=1, cut=False):
    if len(src_dict['candidates']) == 1:
        if datasets == 'zeshel':
            content = 'The prior and context are the same. Predict entity is {}.'.format(src_dict['candidates'][0]['title'])
        else:
            content = 'The prior and context are the same. Predict entity is {}.'.format(src_dict['candidates'][0]['name'])
        return None, content
    
    system_content = "You're an entity disambiguator. I'll give you the description of entity disambiguation and some tips on entity disambiguation, you should pay attention to these textual features:\n\n"
    system_content += instruction_dict[prompt_id]['prompt']

    content = "Now, I'll give you a mention, a context, and a list of candidates entities, the mention will be highlighted with '###' in context.\n\n"
    content += 'Mention:{}\n'.format(src_dict['mention'])

    if ('cut_left_context' in src_dict.keys()) and cut:
        context = src_dict['cut_left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['cut_right_context']
    else:
        context = src_dict['left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['right_context']
    context = context.strip()
    context = ' '.join(context.split())
    content += 'Context:{}\n'.format(context)

    candidates = random.sample(src_dict['candidates'], len(src_dict['candidates']))
    i = 1
    for cand in candidates:
        if datasets == 'zeshel':
            cand_entity = '{}.{}'.format(cand['title'], cand['summary'])
            content += 'Entity {}:{}\n'.format(cand['document_id'], cand_entity)
        else:
            cand_entity = '{}.{}'.format(cand['name'], cand['summary'])
            content += 'Entity {}:{}\n'.format(i, cand_entity)
        i += 1
    content += '\n'

    content += """You need to determine which candidate entity is more likely to be the mention. Please refer to the above example, give your reasons, and finally answer serial number of the entity and the name of the entity. If all candidate entities are not appropriate, you can answer '-1.None'."""

    return system_content, content

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DocFixQA args')
    parser.add_argument('--dataset_name', '-d', type=str, required=True, help="Dataset Name")
    parser.add_argument('--dataset_path', type=str, help="Dataset Path", default=None)
    parser.add_argument('--model_name', '-m', type=str, required=True, help='Model Name')
    parser.add_argument('--exp_name','-e',type=str, required=True, default='test', help='Exp Name')
    parser.add_argument('--model_path','-p',type=str, required=True, help="Path to model")
    parser.add_argument('--func_name','-f',type=str, required=True, help="Function(prompt) used")
    parser.add_argument('--output_key', type=str, help='output json key', default='llm_response')
    parser.add_argument('--instruction_dict', type=str, help='instruction file path', default='prompt/prompt.jsonl')
    parser.add_argument('--COT_pool', type=str, help='COT file path', default='datasets/aida_train_COT_pool.jsonl')
    parser.add_argument('--test', action='store_true', help="if Test", default=False)

    args = parser.parse_args()
    dataset_name = args.dataset_name.lower()
    model_name = args.model_name
    exp_name = args.exp_name
    func_name = args.func_name
    full_flag = False if args.test else True

    pipeline = load_llm(model_name, args.model_path)

    input_file_name = args.dataset_path
    output_file_name = 'result/{}.json'.format(args.exp_name)
    output_key = args.output_key
    
    if func_name == 'context':
        instruction_dict = read_prompt(args.instruction_dict)
        cot_index_dict = read_cot(args.COT_pool)
    elif func_name in ['merge', 'point_wise']:
        instruction_dict = read_prompt(args.instruction_dict)

    if not full_flag:
        with open(input_file_name) as input_f, \
            open(output_file_name, 'w') as output_f:
            for i, line in tqdm(enumerate(input_f)):
                line = json.loads(line.strip())
                if func_name == 'summary':
                    prompt = summary_prompt(line)
                    system_prompt = ''
                elif func_name == 'point_wise':
                    prompt = point_wise_el_prompt(line, instruction_dict, dataset_name, cut=False)
                    system_prompt = ''
                elif func_name == 'category':
                    prompt = category_prompt(line)
                    system_prompt = ''
                elif func_name == 'context':
                    system_prompt, prompt = context_prompt(line, dataset_name, cot_index_dict=cot_index_dict, instruction_dict=instruction_dict, ent_des='summary')
                elif func_name == 'prior':
                    system_prompt, prompt = prior_prompt(line, dataset_name)
                elif func_name == 'merge':
                    system_prompt, prompt = merge_prompt(line, dataset_name, instruction_dict=instruction_dict)
                    if system_prompt == None:
                        response = prompt
                        line[output_key] = response
                        output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
                        continue
                    if len(line['candidates']) == 0:
                        continue

                print('-'*50 + 'Prompt' + '-'*50)
                print(prompt)
                messages = [
                    {"role": "system", "content":system_prompt},
                    {"role": "user", "content": prompt},
                ]
                response = llm_call(messages, model_name, pipeline=pipeline, model_path=args.model_path)
                print('-'*50 + 'Response' + '-'*50)
                print(response)
                line[output_key] = response
                output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
                if i >= 0:
                    break
    else:
        with open(input_file_name) as input_f, \
            open(output_file_name, 'w') as output_f:
            for line in input_f:
                line = json.loads(line.strip())
                if func_name == 'summary':
                    prompt = summary_prompt(line)
                    system_prompt = ''
                elif func_name == 'point_wise':
                    prompt = point_wise_el_prompt(line, instruction_dict, dataset_name, cut=False)
                    system_prompt = ''
                elif func_name == 'category':
                    prompt = category_prompt(line)
                    system_prompt = ''
                elif func_name == 'context':
                    system_prompt, prompt = context_prompt(line, dataset_name, cot_index_dict=cot_index_dict, instruction_dict=instruction_dict)
                elif func_name == 'prior':
                    system_prompt, prompt = prior_prompt(line, dataset_name)
                elif func_name == 'merge':
                    system_prompt, prompt = merge_prompt(line, dataset_name, instruction_dict=instruction_dict)
                    if system_prompt == None:
                        response = prompt
                        line[output_key] = response
                        output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
                        continue
                messages = [
                    {"role": "system", "content":system_prompt},
                    {"role": "user", "content": prompt},
                ]
                try:
                    response = llm_call(messages, model_name, pipeline=pipeline, model_path=args.model_path)
                except:
                    response = ''
                line[output_key] = response
                output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
