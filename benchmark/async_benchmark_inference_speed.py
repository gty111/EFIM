from typing import List, Tuple
import tqdm
import asyncio
import random
import jsonlines
import time
import warnings
import numpy as np
import argparse

from dataclasses import dataclass
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from request_vllm import async_request_openai_completions, RequestFuncInput, RequestFuncOutput


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
parser.add_argument('--num-round', type=int, default=5)
parser.add_argument('--num-user', type=int, default=16)
args = parser.parse_args()

model_path = f'./models/{args.model}'
if 'llama' in model_path:
    model= 'llama' 
elif 'deepseek' in model_path:
    model = 'deepseek'
else:
    assert 0
enhance = 'enhance' in args.model

log_info = ''
log_info += 'eLLM ' if enhance else 'oLLM '
log_info += 'w/EFIM ' if args.use_EFIM else 'w/FIM '
log_info += model
print(log_info)

print(f'num_rounds: {args.num_round} num_user: {args.num_user}')

@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p99_itl_ms: float
    avg_latency_ms: float


def calculate_metrics(
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    tot_input_len: int,
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    latencys: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note : this may inflate the output token count slightly
            output_len = len(
                tokenizer(outputs[i].generated_text,
                          add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            if output_len > 1:
                tpots.append(
                    (outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            latencys.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=tot_input_len,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=tot_input_len / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
        avg_latency_ms=np.mean(latencys) * 1000,
    )

    return metrics, actual_output_lens


def benchmark_each():
    num_round = args.num_round
    num_user = args.num_user
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)

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

    async def user_request(prompt_tot: str, pbar: tqdm.tqdm):
        tot_input_len = 0
        
        tot_prefix_len = len(prompt_tot)//2
        prefix_len = int(tot_prefix_len/(num_round+2))
        suffix = prompt_tot[tot_prefix_len:]
        prefix = prompt_tot[:prefix_len]
        outputs = []
        for round_idx in range(num_round):
            middle = prompt_tot[prefix_len:int(
                tot_prefix_len/(num_round+2)*(round_idx+2))]
            if model == 'deepseek':
                if enhance and prompt_transform:
                    input_text = f'{fim_begin}{prefix}{fim_end}{suffix}{fim_hole}{middle}'
                elif enhance and not prompt_transform:
                    input_text = f'{fim_begin}{prefix+middle}{fim_end}{suffix}{fim_hole}'
                elif not enhance and prompt_transform:
                    input_text = f'{fim_begin}{prefix}{fim_hole}{suffix}{fim_end}{middle}'
                elif not enhance and not prompt_transform:
                    input_text = f'{fim_begin}{prefix+middle}{fim_hole}{suffix}{fim_end}'
                else:
                    assert 0
            elif model == 'llama':
                if prompt_transform:
                    input_text = f'{fim_begin}{prefix}{fim_hole}{suffix}{fim_end}{middle}'
                elif not prompt_transform:
                    input_text = f'{fim_begin}{prefix+middle}{fim_hole}{suffix}{fim_end}'
                else:
                    assert 0
            else:
                assert 0
            tot_input_len += len(tokenizer.encode(input_text))
            request_input = RequestFuncInput(input_text, api_url, len(input_text), max_tokens, model_path,
                                             temperature, top_p, top_k, eos_token_ids, True)
            output = await async_request_openai_completions(
                request_func_input=request_input,
                pbar=pbar)
            outputs.append(output)
        return tot_input_len , outputs

    async def benchmark():
        problems = None
        with open(data_path, 'r') as f:
            problems = list(jsonlines.Reader(f))
        
        problems_filter = []
        for problem in problems:
            input_len = len(tokenizer.encode(problem['prompt']+problem['groundtruth']+problem['right_context']))
            if input_len > 2048 and input_len < 4096:
                problems_filter.append(problem)
        problems = random.sample(problems_filter, num_user)

        start_time = time.perf_counter()
        pbar = tqdm.tqdm(total=len(problems)*num_round)
        tasks = []
        for problem in problems:
            prompt_tot = problem['prompt'] + \
                problem['groundtruth'] + problem['right_context']
            tasks.append(
                asyncio.create_task(user_request(prompt_tot, pbar)))

        outputs = await asyncio.gather(*tasks)
        pbar.close()

        benchmark_duration = time.perf_counter() - start_time

        output_ret = []
        tot_input_len = 0
        for outputs_each in outputs:
            output_ret.extend(outputs_each[1])
            tot_input_len += outputs_each[0]

        metrics, actual_output_lens = calculate_metrics(
            outputs=output_ret, dur_s=benchmark_duration, tokenizer=tokenizer, tot_input_len=tot_input_len)

        print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
        print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
        print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                        benchmark_duration))
        print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
        print("{:<40} {:<10}".format("Total generated tokens:",
                                     metrics.total_output))
        print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                        metrics.request_throughput))
        print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):",
                                    metrics.input_throughput))
        print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                        metrics.output_throughput))
        print("{:<40} {:<10.2f}".format(
            "Avg latency (ms):", metrics.avg_latency_ms))
        print("{s:{c}^{n}}".format(s='Time to First Token', n=50, c='-'))
        print("{:<40} {:<10.2f}".format(
            "Mean TTFT (ms):", metrics.mean_ttft_ms))
        print("{:<40} {:<10.2f}".format("Median TTFT (ms):",
                                        metrics.median_ttft_ms))
        print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
        print("{s:{c}^{n}}".format(s='Time per Output Token (excl. 1st token)',
                                   n=50,
                                   c='-'))
        print("{:<40} {:<10.2f}".format(
            "Mean TPOT (ms):", metrics.mean_tpot_ms))
        print("{:<40} {:<10.2f}".format("Median TPOT (ms):",
                                        metrics.median_tpot_ms))
        print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
        print("{s:{c}^{n}}".format(s='Inter-token Latency', n=50, c='-'))
        print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
        print("{:<40} {:<10.2f}".format(
            "Median ITL (ms):", metrics.median_itl_ms))
        print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
        print("=" * 50)

    asyncio.run(benchmark())


benchmark_each()
