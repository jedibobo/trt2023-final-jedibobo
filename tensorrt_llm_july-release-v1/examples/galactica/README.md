# Galactica


## Setup Env(Install Requirements and Prepare model)
### 获取docker image
```shell
docker pull registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:final_v1
docker run  -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --network host --name hackathon2023  registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:final_v1 bash
```
### install git lfs in docker
```shell
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash 
# no need sudo in docker with root user
apt-get install git-lfs
```
### clone this repository
```shell
git clone https://github.com/jedibobo/trt2023-final-jedibobo.git
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
which add bos_token, eos_token and pad_token when using summarize.py. (In fact, bos_token is not necessary.)

### Install Necessary Python Package
```shell
cd trt2023-final-jedibobo/tensorrt_llm_july-release-v1/examples/galactica
pip install -r requirements.txt
```

### update tensorrt_llm package modified by the author(me)
Otherwise will prompt this error:
```shell
  File "/root/workspace/trt2023-final-jedibobo/tensorrt_llm_july-release-v1/examples/galactica/weight.py", line 9, in <module>
    from tensorrt_llm.models import GALAILMHeadModel
```
Copy Modified Files to:
```shell
cp -r tensorrt_llm_july-release-v1/tensorrt_llm/models/galactica /usr/local/lib/python3.8/dist-packages/tensorrt_llm/models/
cp -r tensorrt_llm_july-release-v1/tensorrt_llm/models/__init__.py /usr/local/lib/python3.8/dist-packages/tensorrt_llm/models/__init__.py
```
**Note**: this method will temporarily solve the module not found issue, I tried to rebuild trt_llm, but got a few problems to be solved.


When you finish these, you can proceed on [building model](#steps-to-build-and-run-summarization).
```shell
Successfully installed tensorrt-llm-0.1.3
```

<a name="Steps to Build and Run Summarization"></a>

## Steps to Build and Run Summarization
### simply run build_run.sh
```shell
sh build_and_run_125m.sh

or

sh build_and_run_1.3b.sh # need download model first
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

Finally, test model

Generation Task:
```shell
python3 run.py --max_output_len=400 --tokenizer_dir galactica-125m/ --engine_dir  trt_engine/galactica-125m/weight_only/1-gpu/
```

Summarization Task:
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
### output of the model(The more params the better!)
The output of Galactica-125m is: 
TRT_LLM output
```
Summary : [[' "I\'ve never seen a person who has never seen a person who has never seen a person who has never seen a person who has never seen a person who has never seen a person who has never seen a person who has never seen a person who has never seen a person who has never seen a person who has never seen a person who has never seen a person who has never seen a person who has never seen a person who has never seen a person who has never seen a person who']]
```
HF_MODEL output (This will improve when the param increases, I have tested 1.3b, it's obviously better.)
```
Summary : [[' "I\'m a very good actor, and I\'m a very good actor."\n\n## Personal life\n\n Best was married to actress and singer-songwriter, Rosco P. Coltrane, in 1948. They had two children, James and Rosco.\n\n## External links\n\n* James Best at IMDb\n* James Best at Find a Grave\n\n']]
```

The output of Galactica-1.3b is: 

TRT_LLM output
```
Summary : [[' "The Dukes of Hazzard" was a hit show for 26 years, and it\'s been a hit for 26 years. It\'s been a hit for 26 years. It\'s been a hit for 26 years. It\'s been a hit for 26 years. It\'s been a hit for 26 years. It\'s been a hit for 26 years. It\'s been a hit']]
```
HF_MODEL output (This will improve when the param increases, I have tested 1.3b, it's obviously better.)
```
Summary : [[' "The Dukes of Hazzard" was a hit show for 26 years, and James Best was a star. But he died in 2015, and his character will live on in reruns.']]
```

The output of Galactica-6.7b(**unable to run on A10**) is: 

TRT_LLM output
```
Summary : [[' James Best, best known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV\'s "The Dukes of Hazzard," died Monday after a brief illness. He was 88. Best died in hospice in Hickory, North Carolina, of complications from pneumonia, said Steve Latshaw, a longtime friend and Hollywood colleague. Although he\'d been a busy actor for decades in theater and in Hollywood,']]
```
HF_MODEL output (This will improve when the param increases, I have tested 1.3b, it's obviously better.)
```
Summary : [[' James Best, best known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV\'s "The Dukes of Hazzard," died Monday after a brief illness. He was 88. Best died in hospice in Hickory, North Carolina, of complications from pneumonia, said Steve Latshaw, a longtime friend and Hollywood colleague. Although he\'d been a busy actor for decades in theater and in Hollywood,']]
```


### Metrics
125M results are shown below:
for 6.7B model(rouge1 rouge1 : 19.116646730129816), it almost touches limit of rouge1>20.
```
[09/19/2023-12:52:13] [TRT-LLM] [I] TensorRT-LLM (total latency: 3.6322696208953857 sec)
[09/19/2023-12:52:13] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[09/19/2023-12:52:13] [TRT-LLM] [I]   rouge1 : 10.277576407234886
[09/19/2023-12:52:13] [TRT-LLM] [I]   rouge2 : 1.6005120808715905
[09/19/2023-12:52:13] [TRT-LLM] [I]   rougeL : 8.08101934142795
[09/19/2023-12:52:13] [TRT-LLM] [I]   rougeLsum : 9.833098488425106
[09/19/2023-12:52:13] [TRT-LLM] [I] Hugging Face (total latency: 10.479159355163574 sec)
[09/19/2023-12:52:13] [TRT-LLM] [I] HF beam 0 result
[09/19/2023-12:52:14] [TRT-LLM] [I]   rouge1 : 11.029703712283155
[09/19/2023-12:52:14] [TRT-LLM] [I]   rouge2 : 1.7302256076234355
[09/19/2023-12:52:14] [TRT-LLM] [I]   rougeL : 8.42338520559755
[09/19/2023-12:52:14] [TRT-LLM] [I]   rougeLsum : 10.437959123452638
```
1.3B
```
[09/20/2023-09:23:41] [TRT-LLM] [I] TensorRT-LLM (total latency: 14.450116634368896 sec)
[09/20/2023-09:23:41] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[09/20/2023-09:23:41] [TRT-LLM] [I]   rouge1 : 15.82523484186119
[09/20/2023-09:23:41] [TRT-LLM] [I]   rouge2 : 3.0146271394965547
[09/20/2023-09:23:41] [TRT-LLM] [I]   rougeL : 12.201330312061408
[09/20/2023-09:23:41] [TRT-LLM] [I]   rougeLsum : 13.835589369518091
[09/20/2023-09:23:41] [TRT-LLM] [I] Hugging Face (total latency: 19.247478008270264 sec)
[09/20/2023-09:23:41] [TRT-LLM] [I] HF beam 0 result
[09/20/2023-09:23:41] [TRT-LLM] [I]   rouge1 : 16.849348629816337
[09/20/2023-09:23:41] [TRT-LLM] [I]   rouge2 : 3.403001301754608
[09/20/2023-09:23:41] [TRT-LLM] [I]   rougeL : 12.892710146409058
[09/20/2023-09:23:41] [TRT-LLM] [I]   rougeLsum : 14.695898263141338
```

#### enable fmha for Galactica-125M and Galactica-1.3B
125M
```
[09/20/2023-01:59:00] [TRT-LLM] [I] TensorRT-LLM (total latency: 3.344346761703491 sec)
[09/20/2023-01:59:00] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[09/20/2023-01:59:00] [TRT-LLM] [I]   rouge1 : 10.277576407234886
[09/20/2023-01:59:00] [TRT-LLM] [I]   rouge2 : 1.6005120808715905
[09/20/2023-01:59:00] [TRT-LLM] [I]   rougeL : 8.08101934142795
[09/20/2023-01:59:00] [TRT-LLM] [I]   rougeLsum : 9.833098488425106
[09/20/2023-01:59:00] [TRT-LLM] [I] Hugging Face (total latency: 10.588122367858887 sec)
[09/20/2023-01:59:00] [TRT-LLM] [I] HF beam 0 result
[09/20/2023-01:59:00] [TRT-LLM] [I]   rouge1 : 11.029703712283155
[09/20/2023-01:59:00] [TRT-LLM] [I]   rouge2 : 1.7302256076234355
[09/20/2023-01:59:00] [TRT-LLM] [I]   rougeL : 8.42338520559755
[09/20/2023-01:59:00] [TRT-LLM] [I]   rougeLsum : 10.437959123452638
```

1.3B
```
[09/20/2023-02:15:37] [TRT-LLM] [I] TensorRT-LLM (total latency: 13.790514469146729 sec)
[09/20/2023-02:15:37] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[09/20/2023-02:15:37] [TRT-LLM] [I]   rouge1 : 16.17296647271511
[09/20/2023-02:15:37] [TRT-LLM] [I]   rouge2 : 3.1515046849233523
[09/20/2023-02:15:37] [TRT-LLM] [I]   rougeL : 12.345827479167633
[09/20/2023-02:15:37] [TRT-LLM] [I]   rougeLsum : 14.257476328986169
[09/20/2023-02:15:37] [TRT-LLM] [I] Hugging Face (total latency: 19.12988781929016 sec)
[09/20/2023-02:15:37] [TRT-LLM] [I] HF beam 0 result
[09/20/2023-02:15:37] [TRT-LLM] [I]   rouge1 : 16.849348629816337
[09/20/2023-02:15:37] [TRT-LLM] [I]   rouge2 : 3.403001301754608
[09/20/2023-02:15:37] [TRT-LLM] [I]   rougeL : 12.892710146409058
[09/20/2023-02:15:37] [TRT-LLM] [I]   rougeLsum : 14.695898263141338
```
You can see there is not much performance drop compared to HF resultss.

#### enable fmha and weight_only
125m
```
[09/20/2023-03:23:09] [TRT-LLM] [I] TensorRT-LLM (total latency: 2.7617087364196777 sec)
[09/20/2023-03:23:09] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[09/20/2023-03:23:09] [TRT-LLM] [I]   rouge1 : 10.813010399725236
[09/20/2023-03:23:09] [TRT-LLM] [I]   rouge2 : 1.5951352415935847
[09/20/2023-03:23:09] [TRT-LLM] [I]   rougeL : 8.516456604041817
[09/20/2023-03:23:09] [TRT-LLM] [I]   rougeLsum : 9.859581700840726
[09/20/2023-03:23:09] [TRT-LLM] [I] Hugging Face (total latency: 10.66635537147522 sec)
[09/20/2023-03:23:09] [TRT-LLM] [I] HF beam 0 result
[09/20/2023-03:23:10] [TRT-LLM] [I]   rouge1 : 11.029703712283155
[09/20/2023-03:23:10] [TRT-LLM] [I]   rouge2 : 1.7302256076234355
[09/20/2023-03:23:10] [TRT-LLM] [I]   rougeL : 8.42338520559755
[09/20/2023-03:23:10] [TRT-LLM] [I]   rougeLsum : 10.437959123452638
```
1.3B
```
[09/20/2023-03:19:07] [TRT-LLM] [I] TensorRT-LLM (total latency: 9.21572732925415 sec)
[09/20/2023-03:19:07] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[09/20/2023-03:19:07] [TRT-LLM] [I]   rouge1 : 14.86201096912762
[09/20/2023-03:19:07] [TRT-LLM] [I]   rouge2 : 2.8026336944750434
[09/20/2023-03:19:07] [TRT-LLM] [I]   rougeL : 12.546048137123414
[09/20/2023-03:19:07] [TRT-LLM] [I]   rougeLsum : 13.450615751879615
[09/20/2023-03:19:07] [TRT-LLM] [I] Hugging Face (total latency: 19.304476976394653 sec)
[09/20/2023-03:19:07] [TRT-LLM] [I] HF beam 0 result
[09/20/2023-03:19:07] [TRT-LLM] [I]   rouge1 : 16.849348629816337
[09/20/2023-03:19:07] [TRT-LLM] [I]   rouge2 : 3.403001301754608
[09/20/2023-03:19:07] [TRT-LLM] [I]   rougeL : 12.892710146409058
[09/20/2023-03:19:07] [TRT-LLM] [I]   rougeLsum : 14.695898263141338
```
When both weight only and FMHA is enabled, the conclusion is on A10 GPU, the accuracy loss is a little significant(rough1 diff >1)

### Speed-up Ratio
For Galactica-125M, the speed-up is 2.885 on A10 Ali cloud machine.
```
TensorRT-LLM (total latency: 3.6322696208953857 sec)
Hugging Face (total latency: 10.479159355163574 sec)
```

For Galactica-1.3B, the speed-up is 1.314 on A10 Ali cloud machine.
```
TensorRT-LLM (total latency: 14.435588836669922 sec)
Hugging Face (total latency: 18.968894243240356 sec)
```

#### Enable fmha
For Galactica-125M, the speed-up is 3.166 on A10 Ali cloud machine, when fmha is enabled.
```
TensorRT-LLM (total latency: 3.344346761703491 sec)
Hugging Face (total latency: 10.588122367858887 sec)
```
For Galactica-1.3B, the speed-up is 1.387 on A10 Ali cloud machine, when fmha is enabled.
```
TensorRT-LLM (total latency: 13.790514469146729 sec)
Hugging Face (total latency: 19.12988781929016 sec)
```

**Conclusion:Compared to No FMHA, there is a subtle improvement without much performance drop or accuracy loss.**

#### Enable fmha and weight_only
For Galactica-125M, the speed-up is 3.862 on A10 Ali cloud machine, when fmha is enabled.
```
TensorRT-LLM (total latency: 2.7617087364196777 sec)
Hugging Face (total latency: 10.66635537147522 sec)
```

For Galactica-1.3B, the speed-up is 2.095 on A10 Ali cloud machine, when fmha is enabled.
```
TensorRT-LLM (total latency: 9.21572732925415 sec)
Hugging Face (total latency: 19.304476976394653 sec)
```

### Future Work
Currently, I am not capable of and lack time of debugging the way as those scripts in [**tests**](../../tests/model/test_gpt_e2e.py) dir 