PORT=65511
HOST=0.0.0.0

CMD="vllm serve ./models/$MODEL --host $HOST --port $PORT --enforce-eager
        --disable-log-requests --trust-remote-code"

echo $CMD
$CMD