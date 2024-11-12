# GPT-2 (124M) repro on FineWeb
# 124M parameter model on 10B tokens
# => 6 * 124e6 * 10e9 = 7.44e18 ~= 7e18 capability model
# 18,865 steps of 524,288 tokens/step
# on 8X A100 80GB SXM ($14/hr) steps in ~300ms/iter
# => training time 18,865 * 300ms = 94.3 min ~= $20

make train_gpt2cu USE_CUDNN=1
out_dir="/home/cmurphy/experiments/gencity/train-nyc-llm"
done_file="$out_dir/DONE_00018865"

# in case the training stalls or crashes, loop to resume (-y 1)
while true; do

    # exit condition is that optimization has finished
    if [ -f "$done_file" ]; then
        echo "File $done_file exists. Exiting the loop."
        break
    fi

    # run python dev/data/fineweb.py --version 10B to prepro data
    # run python dev/data/hellaswag.py to prepro hellaswag eval
    # -i: input training data
    # -j: input eval data
    # -o: output model directory
    # -v: val_loss_every
    # -s: sample_every
    # -g: number of tokens sampled
    # -h: number hella swag samples
    # -b: batch size
    # -t: context size
    # -d: desired batch size (after gradient accumulation)
    # -r: recompute (not sure)
    # -z: zero optimization stage
    # -c: weight decay
    # -l: learning rate
    # -q: learning rate decay
    # -u: warmup steps
    # -n: checkpoint_every
    # -y: resume training
    # -e: input .bin filename or descriptor



    mpirun -np 1 /home/cmurphy/repos/llm.c/train_gpt2cu \
                -i "/home/cmurphy/datasets/gencity/nyc-tokens-gcp/nyc-tokens-*.bin" \
                -j "/home/cmurphy/datasets/gencity/nyc-tokens-gcp/nyc-val-tokens.bin" \
                -tp "/home/cmurphy/datasets/gencity/nyc-tokens-gcp/tokenizer.bin" \
                -o $out_dir \
                -v 250 -s 200 -g 144 \
                -h 0 \
                -b 24 -t 2048\
                -d 786432 \
                -r 0 \
                -z 1 \
                -c 0.1 \
                -l 0.0006 \
                -q 0.0 \
                -u 700 \
                -n 1000 \
                -y 1 \
		-x 50000\
                -e "gpt3:c768" > $out_dir/train.log 2>&1

    sleep 1
    break
done
