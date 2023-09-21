rm -rf ./c-model/
rm -rf ./trt_engine/
python3 hf_galactica_convert.py -i galactica-1.3b \
                                -o ./c-model/galactica-1.3b/fp16 \
                                -i_g 1 -weight_data_type fp16 2>&1 | tee hf_convert_galai_1.3b_ft.log

rm -rf ./c-model/galactica-1.3b/fp16/*.bin

python3 build.py --model_dir=./c-model/galactica-1.3b/fp16/1-gpu \
                 --hf_model_dir=./galactica-1.3b/ \
                 --max_batch_size 8 \
                 --dtype float16 \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16 \
                 --use_layernorm_plugin float16\
                 --max_input_len 1024 \
                 --max_output_len 400 \
                 --world_size 1 \
                 --output_dir trt_engine/galactica-1.3b/fp16/1-gpu \
                 --pre_norm \
                 --hidden_act gelu 2>&1 | tee build.log


python3 summarize.py --engine_dir trt_engine/galactica-1.3b/fp16/1-gpu \
                     --test_hf \
                     --batch_size 1 \
                     --test_trt_llm \
                     --hf_model_location ./galactica-1.3b/ \
                     --data_type fp16 \
                     --tensorrt_llm_rouge1_threshold=20 