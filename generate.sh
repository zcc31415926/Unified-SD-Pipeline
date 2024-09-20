python ./scripts/generate.py \
    --model_type sd1 \
    --prompt "a professional photograph of an astronaut riding a horse" \
    --num_generation 4 \
    --out_dir ./logs \
    --batch_size 1 \
    --out_size 512 \
    --guidance_scale 7.5 \
    --fp16 \
    --device cuda

