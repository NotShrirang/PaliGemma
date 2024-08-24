#!/bin/bash

MODEL_PATH="$HOME/projects/paligemma-weights/paligemma-3b-pt-224"
PROMPT="This building is "
IMAGE_PATH="louvre.jpg"
MAX_TOKENS_TO_GENERATE=100
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="True"

python inference.py \
    --model_path $MODEL_PATH \
    --prompt "$PROMPT" \
    --image_path $IMAGE_PATH \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \