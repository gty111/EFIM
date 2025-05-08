# Set up Environment

We need conda to set up environment. Please install conda before executing the following instructions.
```
source scripts/setup.sh
```
After setting up environment, you should execute the instructions under conda env `euro_par_artifact`.
Please make sure models (`llama`, `llama-enhance`, `deepseek`, `deepseek-enhance`) are properly downloaded under `models`.

# Launch vLLM

## w/o prefix caching
```
export MODEL=llama # choose from [llama, llama-enhance, deepseeek, deepseek-enhance]
./scripts/launch_server.sh
```
## w/ prefix caching
```
export MODEL=llama # choose from [llama, llama-enhance, deepseeek, deepseek-enhance]
./scripts/launch_prefix_server.sh
```

# Evaluate Infilling and Subtoken Generation Ability

**Launch vLLM before running following commands**

## HumanEval 

### w/FIM
```
python benchmark/async_benchmark_humaneval.py --model llama
```
### w/EFIM
```
python benchmark/async_benchmark_humaneval.py --model llama --use-EFIM
```

## CCEval

### w/FIM
```
python benchmark/async_benchmark_cceval.py --model llama
```

### w/EFIM
```
python benchmark/async_benchmark_cceval.py --model llama --use-EFIM
```

# Evaluate Inference Speedup

**Launch vLLM before running following commands**

### w/FIM
```
python benchmark/async_benchmark_inference_speed.py --model llama --num-round 5 --num-user 16 
```

### w/EFIM
```
python benchmark/async_benchmark_inference_speed.py --model llama-enhance --num-round 5 --num-user 16 --use-EFIM
```