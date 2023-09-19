# Galactica


## Install Requirements and Prepare model
### install git lfs in docker
```shell
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash # no need sudo in docker with root user
apt-get install git-lfs
```

### clone model from huggingface for 125m model
```shell
cd trt2023-final-jedibobo/tensorrt_llm_july-release-v1/examples/galactica
git lfs install # make sure git lfs is installed using apt
git clone https://huggingface.co/facebook/galactica-125m
```
Then, change the tokenizer_config.json file with the following:
```json
{
  "name_or_path": "/content/tokenizer",
  "special_tokens_map_file": "/content/tokenizer/special_tokens_map.json",
  "tokenizer_class": "PreTrainedTokenizerFast",
  "bos_token": {
    "__type": "AddedToken",
    "content": "<s>",
    "lstrip": false,
    "normalized": true,
    "rstrip": false,
    "single_word": false
  },
  "eos_token": {
  "__type": "AddedToken",
  "content": "</s>",
  "lstrip": false,
  "normalized": true,
  "rstrip": false,
  "single_word": false
  },
  "pad_token": null
}
```
which add bos_token, eos_token and pad_token when using summarize.py.

### Install Necessary Python Package
```shell
pip install -r requirements.txt
```

### update tensorrt_llm package modified by me
Otherwise will prompt this error:
```shell
  File "/root/workspace/trt2023-final-jedibobo/tensorrt_llm_july-release-v1/examples/galactica/weight.py", line 9, in <module>
    from tensorrt_llm.models import GALAILMHeadModel
```
Update the TRT_LLM pip package by:
```shell
cd trt2023-final-jedibobo/tensorrt_llm_july-release-v1/
pip install -e .
```
when you see the following lines in CLI, you can proceed on [building model](#steps-to-build-and-run-summarization).
```shell
Installing collected packages: tensorrt-llm
  Attempting uninstall: tensorrt-llm
    Found existing installation: tensorrt-llm 0.1.3
    Uninstalling tensorrt-llm-0.1.3:
      Successfully uninstalled tensorrt-llm-0.1.3
  Running setup.py develop for tensorrt-llm
Successfully installed tensorrt-llm-0.1.3
```

<a name="Steps to Build and Run Summarization"></a>

## Steps to Build and Run Summarization
### simply run build_run.sh
```shell
sh build_and_run_125m.sh
```

### build 125m model
```shell
## ft format loading weight has problems saving with all zero weights, this process just provice a config.ini file that contains necessary parameters when building models
rm -rf ./c-model/
rm -rf ./trt_engine/
python3 hf_galactica_convert.py -i galactica-125m \
                                -o ./c-model/galactica-125m/fp16 \
                                -i_g 1 -weight_data_type fp16 > hf_convert_galai_125m_ft.log 2>&1
```
The previous line may generate a warning 'Some weights of OPTForCausalLM were not initialized from the model checkpoint at galactica-125m and are newly initialized: ['lm_head.weight']', I discover it is a problem of original 125m model, I save it again using this help [page](https://colab.research.google.com/drive/1hjnB9VBMnbVIJiTNdQWZbL0yIAqF75WZ?usp=sharing#scrollTo=m--OZ0l8bCQQ) and got no more **'newly initialized weight'** warning. 

Then, build the model using tensorrt_llm with FP16 precision.
```shell
python3 build.py --model_dir=./c-model/galactica-125m/fp16/1-gpu \
                 --hf_model_dir=./galactica-125m/ \
                 --max_batch_size 8 \
                 --dtype float16 \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16 \
                 --use_layernorm_plugin float16\
                 --max_input_len 1024 \
                 --max_output_len 400 \
                 --world_size 1 \
                 --output_dir trt_engine/galactica-125m/fp16/1-gpu \
                 --pre_norm \
                 --hidden_act gelu > build.log 2>&1
```

some keywords to mind are:
- pre_norm is True for all Galactica models.
- hidden_act gelu for all Galactica models.

Finally,test model
```shell
python3 summarize.py --engine_dir trt_engine/galactica-125m/fp16/1-gpu \
                     --test_hf \
                     --batch_size 1 \
                     --test_trt_llm \
                     --hf_model_location ./galactica-125m/ \
                     --data_type fp16 \
                     --tensorrt_llm_rouge1_threshold=20 
```

## Result
### output of the model
The output of llm is: (How ridiculous...)
TRT_LLM output
```
Summary : [[' "Crive, "Crive, "Crive, "Crive, "Crive, "Crive, "Crive, "Crive, "Crive, "Crive, "Crive, "Crive, "Crive, "Crive, "Crive, "Crive, "Cri']]
```
HF_MODEL output (This will improve when the param increases, I have tested 1.3b, it's obviously better.)
```
Summary : [[' "I\'m a very good actor, and I\'m a very good actor."\n\n## Personal life\n\n Best was married to actress and singer-songwriter, Rosco P. Coltrane, in 1948. They had two children, James and Rosco.\n\n## External links\n\n* James Best at IMDb\n* James Best at Find a Grave\n\n']]
```


### Metrics
```
[09/19/2023-08:45:48] [TRT-LLM] [I] TensorRT-LLM (total latency: 3.5248897075653076 sec)
[09/19/2023-08:45:48] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[09/19/2023-08:45:49] [TRT-LLM] [I]   rouge1 : 2.4951468751395955
[09/19/2023-08:45:49] [TRT-LLM] [I]   rouge2 : 0.0
[09/19/2023-08:45:49] [TRT-LLM] [I]   rougeL : 2.476452405491556
[09/19/2023-08:45:49] [TRT-LLM] [I]   rougeLsum : 2.522421130912537
[09/19/2023-08:45:49] [TRT-LLM] [I] Hugging Face (total latency: 10.462498426437378 sec)
[09/19/2023-08:45:49] [TRT-LLM] [I] HF beam 0 result
[09/19/2023-08:45:49] [TRT-LLM] [I]   rouge1 : 11.029703712283155
[09/19/2023-08:45:49] [TRT-LLM] [I]   rouge2 : 1.7302256076234355
[09/19/2023-08:45:49] [TRT-LLM] [I]   rougeL : 8.42338520559755
[09/19/2023-08:45:49] [TRT-LLM] [I]   rougeLsum : 10.437959123452638
```
### Speed-up Ratio




Currently, I am not capable of and lack time of debugging like those scripts in [**tests**](../../tests/model/test_gpt_e2e.py) dir 