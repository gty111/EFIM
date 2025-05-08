import tqdm
import asyncio
import random
import argparse

from human_eval_infilling.data import write_jsonl, read_problems
from human_eval_infilling.evaluation import evaluate_functional_correctness
from transformers import AutoTokenizer

from request_vllm import async_request_openai_completions, RequestFuncInput

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['deepseek', 'deepseek-enhance', 'llama', 'llama-enhance'], default='llama')
parser.add_argument('--use-EFIM', action='store_true')
parser.add_argument('--num-samples', help='number of samples per task', type=int, default=1)
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

log_info = 'humanEval '
log_info += 'eLLM ' if enhance else 'oLLM '
log_info += 'w/EFIM ' if args.use_EFIM else 'w/FIM '
log_info += model
print(log_info)

def benchmark_each(benchmark_name):
    print('-'*10)
    print(benchmark_name)
    
    max_tokens = args.max_tokens
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    port = args.port
    prompt_transform = args.use_EFIM
    num_samples_per_task = args.num_samples

    random.seed(0)
    api_url = f'http://{args.host}:{port}/v1/completions'
    sample_file = f"log/{benchmark_name}-enhance.jsonl" if enhance else f"log/{benchmark_name}.jsonl"
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
        problems = read_problems(benchmark_name=benchmark_name)

        pbar = tqdm.tqdm(total=num_samples_per_task*len(problems))
        samples = []
        tasks = []
        for _ in range(num_samples_per_task):
            for task_id in problems:
                prompt=problems[task_id]["prompt"]
                suffix=problems[task_id]["suffix"]
                idx = random.randint(1,len(prompt)-1) 
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
                                                    temperature, top_p, top_k, eos_token_ids, False)
                tasks.append(
                    asyncio.create_task(async_request_openai_completions(
                        request_func_input=request_input,
                        pbar=pbar)))
                samples.append(dict(task_id=task_id, prompt=input_text ,completion=''))
        completions = await asyncio.gather(*tasks)
        for idx,sample_each in enumerate(samples):
            assert completions[idx].success
            sample_each['completion'] = completions[idx].generated_text
        write_jsonl(sample_file, samples)

    asyncio.run(benchmark())

    results = evaluate_functional_correctness(benchmark_name,sample_file)
    k = f'pass@{num_samples_per_task}'
    print(f'{k}: {results[k]*100}')

# benchmark_names = ['single-line','multi-line','random-span', 'random-span-light', 'test']
for benchmark_name in ['random-span','single-line','multi-line']:
        benchmark_each(benchmark_name=benchmark_name)