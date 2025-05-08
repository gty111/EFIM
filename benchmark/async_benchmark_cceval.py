import tqdm
import asyncio
import random
import jsonlines
import argparse

from transformers import AutoTokenizer

from fuzzywuzzy import fuzz

from Euro_Par_artifact.benchmark.request_vllm import async_request_openai_completions, RequestFuncInput

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['deepseek', 'deepseek-enhance', 'llama', 'llama-enhance'], default='llama')
parser.add_argument('--cceval-path', default='./data/line_completion.jsonl')
parser.add_argument('--use-EFIM', action='store_true')
parser.add_argument('--host', type=str, default='0.0.0.0')
parser.add_argument('--port', type=int, default=65511)
parser.add_argument('--max-tokens', type=int, default=128)
parser.add_argument('--temperature', type=float, default=0.2)
parser.add_argument('--top-k', type=int, default=20)
parser.add_argument('--top-p', type=float, default=0.95)
args = parser.parse_args()

model_path = f'./models/{args.model}'
if 'llama' in model_path:
    model= 'llama' 
elif 'deepseek' in model_path:
    model = 'deepseek'
else:
    assert 0
enhance = 'enhance' in args.model

log_info = 'cceval '
log_info += 'eLLM ' if enhance else 'oLLM '
log_info += 'w/EFIM ' if args.use_EFIM else 'w/FIM '
log_info += model
print(log_info)

def cal_edit_sim(references, hypotheses):
    total = len(references)
    edit_sim = 0.0
    for pred, gt in zip(hypotheses, references):
        pred = pred.strip()
        gt = gt.strip()
        edit_sim += fuzz.ratio(pred, gt)
    return edit_sim / total

def benchmark_each():
    data_path = args.cceval_path
    
    max_tokens = args.max_tokens
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    port = args.port
    host = args.host
    prompt_transform = args.use_EFIM
    
    api_url = f'http://{host}:{port}/v1/completions'
    random.seed(0)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if model == 'deepseek':
        fim_begin = '<｜fim▁begin｜>'
        fim_hole = '<｜fim▁hole｜>'
        fim_end = '<｜fim▁end｜>'
    elif model == 'llama':
        fim_begin = '<fim_prefix>'
        fim_hole = '<fim_suffix>'
        fim_end = '<fim_middle>'
    else:
        assert 0
    
    if model == 'deepseek':
        eos_token_ids = [32027] if enhance else [tokenizer.eos_token_id]
    elif model == 'llama':
        eos_token_ids = [128000] if not enhance else [tokenizer.eos_token_id]
    else:
        assert 0 

    async def benchmark():
        problems = None
        with open(data_path, 'r') as f:
            problems = list(jsonlines.Reader(f))
        
        problems = [problem for problem in problems if len(tokenizer.encode(problem['prompt']+problem['groundtruth']+problem['right_context'])) < 2048]

        pbar = tqdm.tqdm(total=len(problems))
        samples = []
        tasks = []
        for problem in problems:
            prompt_tot = problem['prompt'] + problem['groundtruth'] + problem['right_context']
            len_groundtruth = random.randint(16,128)
            begin_groundtruth = random.randint(32,len(prompt_tot) - 160)
            prompt = prompt_tot[:begin_groundtruth]
            suffix = prompt_tot[begin_groundtruth+len_groundtruth:]
            idx = random.randint(1,begin_groundtruth-1)
            problem['groundtruth'] = prompt_tot[begin_groundtruth:begin_groundtruth+len_groundtruth]
            if model == 'deepseek':
                if enhance and prompt_transform:
                    input_text = f'{fim_begin}{prompt[:idx]}{fim_end}{suffix}{fim_hole}{prompt[idx:]}'
                elif enhance and not prompt_transform:
                    input_text = f'{fim_begin}{prompt}{fim_end}{suffix}{fim_hole}'
                elif not enhance and not prompt_transform:
                    input_text = f'{fim_begin}{prompt}{fim_hole}{suffix}{fim_end}'
                elif not enhance and prompt_transform:
                    input_text = f'{fim_begin}{prompt[:idx]}{fim_hole}{suffix}{fim_end}{prompt[idx:]}'
                else:
                    assert 0
            elif model == 'llama':
                if not prompt_transform:
                    input_text = f'{fim_begin}{prompt}{fim_hole}{suffix}{fim_end}'
                elif prompt_transform:
                    input_text = f'{fim_begin}{prompt[:idx]}{fim_hole}{suffix}{fim_end}{prompt[idx:]}'
                else:
                    assert 0
            else:
                assert 0
            request_input = RequestFuncInput(input_text,api_url,len(input_text),max_tokens,model_path,
                                            temperature,top_p, top_k, eos_token_ids, False)
            tasks.append(
                asyncio.create_task(async_request_openai_completions(
                    request_func_input=request_input,
                    pbar=pbar)))
            samples.append(dict(prompt=input_text, groundtruth=problem['groundtruth'] ,completion=''))
                
        completions = await asyncio.gather(*tasks)
        pbar.close()
        em = 0
        refs, hyps = [], []
        for idx,sample_each in enumerate(samples):
            assert completions[idx].success
            sample_each['completion'] = completions[idx].generated_text
            if sample_each['groundtruth'] == sample_each['completion']:
                em += 1
            refs.append(sample_each['groundtruth'])
            hyps.append(sample_each['completion'])
        print(f'ES: {cal_edit_sim(refs, hyps)}')
        print(f'EM: {em/len(samples)*100}')
        
            
    asyncio.run(benchmark())
    
benchmark_each()