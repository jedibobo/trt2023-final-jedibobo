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
### build 125m model
```shell
## ft format loading weight has problems saving with all zero weights, this process just provice a config.ini file that contains necessary parameters when building models
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