# 关于复赛与TensorRT-LLM
&emsp;大语言模型是计算机行业未来的重要方向，英伟达希望借助复赛的机会，加强开发团队与开发者的交流，让开发者快速上手英伟达即将正式推出的大语言模型推理加速库TensorRT-LLM，并能在未来的工作中熟练运用。

&emsp;TensorRT-LLM是对TensorRT的再封装。它改善了TensorRT模型的手工搭建方式，引入了plugin提高推理性能，并加入了大量新功能。  
  + 虽然取的名字提到LLM（Large Language Model，大语言模型），TensorRT-LLM可以用来搭建任意AI模型。
  + TensorRT-LLM现在没有ONNX parser，所以不能走ONNX workflow，必须手工搭建模型。
  + 大家拿到的TensorRT-LLM只是一个非公开的预览版本。在使用过程中可能会遇到一些问题，比如没有支持完善的feature。英伟达正在开发完善它。

## 评分标准
+ TensorRT-LLM 试用送分题：为了鼓励选手试用TensorRT-LLM，无论选手选择 TensorRT 还是 TensorRT-LLM 来做优化，完成送分题都能得分
  - 请在报告中写出 /root/workspace/tensorrt_llm_july-release-v1/examples/gpt/README 里面 “Single node, single GPU” 部分如下命令的输出（10分）[模型为gpt2-medium](https://huggingface.co/gpt2-medium)
    - python3 run.py --max_output_len=8 
  - 请在报告中写出 /root/workspace/tensorrt_llm_july-release-v1/examples/gpt/README 里面 “Summarization using the GPT model” 部分如下命令的rouge 分数（10分）[模型为gpt2-medium](https://huggingface.co/gpt2-medium)
    - python3 summarize.py --engine_dirtrt_engine/gpt2/fp16/1-gpu --test_hf  --batch_size1  --test_trt_llm  --hf_model_location=gpt2 --check_accuracy --tensorrt_llm_rouge1_threshold=14

+ 主要得分：
  - 选题得分：请从以下题目选择1个来做，也可以在选2或3的同时叠加4。选TensorRT和选TensorRT-LLM的评分区别仅在选题上。前者最高分为30，后者最高分为100（如果选了2+4或3+4）。
    1. 用TensorRT优化模型（30分）
    2. 用TensorRT-LLM实现新模型（50分）。满足以下条件可给满分：
         - 仿照examples的代码组织形式，完成模型搭建，并可顺利输出文本（实现weight.py/build.py/run.py）
         - 通过摘要任务，可评估原始模型与加速模型的rouge score（实现summarize.py）
         - 为模型写好README.md
         - 与现有模型相比，新模型有一定难度（比如有特殊的结构或算子）
    3. 用TensorRT-LLM优化examples目录下的某个现有模型（50分）。满足以下条件可给满分：
         - 在保持精度的同时，显著提高了性能
         - 为算子提供了更高效率的实现
    4. 为TensorRT-LLM添加新feature，或者在模型上启用了现有feature（50分）  
      这里的feature是指为了更有效率地运行模型，在模型上采用的通用工具。比如现有feature包括int8 KV-cache，inflight batching，SmoothQuant等（这些feature可能在某些现有模型上不能正常工作）。你可以添加新的feature，比如新的量化方法，新的sampling方法等，并在现有模型或新模型中实现。视实现难度与工作量给分。  
      例如，以下为英伟达正在进行的feature开发工作，计划在9月发布：  
         - 在GPT-NEOX和LLaMA上实现GPTQ
         - 在Bloom上实现SmoothQuant  

  - 代码得分：代码干净，逻辑清晰（30分）
  - 报告得分：报告完整，可读性好，对 TensorRT 或 TensorRT-LLM 学习者有良好的参考价值（30分）

+ 附加得分
  - 独立开发了 Plugin 或开发 CUDA 代码（10分）
    - Plugin开发可使用 [OpenAI Triton](https://github.com/openai/triton)。如需在 TensorRT-LLM 中使用，请参考 TensorRT-LLM docker 中 /root/workspace/tensorrt_llm_july-release-v1/examples/openai_triton。
  - 用Nsight Systems/Nsight Compute进行了Profiling，并进行了针对性优化（10分）
  - 提交与开发过程相关的bug，并得到导师确认。提交多个bug不重复加分。（10分）

+ 初赛得分
  - 初赛原始得分除以100取整计入复赛。


## 总述
&emsp;本项目主要贡献在于：使用Nvidia发布的TensorRT_LLM工具，对Galactica系列模型([paper](https://arxiv.org/abs/2211.09085)) ([website](https://galactica.org/)) ([official-demo](https://galactica.org/explore/)) ([huggingface-model](https://huggingface.co/facebook/galactica-6.7b))进行推理优化，具体选题内容为：2.用TensorRT-LLM实现新模型。

### Galactica模型简单介绍
```
    Galactica models are trained on a large corpus comprising more than 360 millions in-context citations and over 50 millions of unique references normalized across a diverse set of sources. This enables Galactica to suggest citations and help discover related papers.
```
&emsp;Galactica大型语言模型（LLM）正在用数百万条学术内容进行训练。它的目的是帮助研究界更好地管理"信息爆炸"。Galactica是由Meta AI与Papers with Code合作开发的。该团队认为信息过载是科学进步的一个主要障碍。"研究人员被埋没在大量的论文中，越来越无法区分有意义的和无意义的"。简单来讲，就是**聚合有意义的内容**，简直是我等科研狗的福音。

### Galactica模型的特点
```
Architecture

Galactica uses a Transformer architecture in a decoder-only setup (Vaswani et al., 2017), with the following
modifications:
• GeLU Activation - we use GeLU activations for all model sizes (Hendrycks and Gimpel, 2016).
• Context Window - we use a 2048 length context window for all model sizes.
• No Biases - following PaLM, we do not use biases in any of the dense kernels or layer norms (Chowdhery
et al., 2022).
• Learned Positional Embeddings - we use learned positional embeddings for the model. We experimented
with ALiBi at smaller scales but did not observe large gains, so we did not use it (Press et al.,
2021).
• Vocabulary - we construct a vocabulary of 50k tokens using BPE (Sennrich et al., 2015). The
vocabulary was generated from a randomly selected 2% subset of the training data.
```
总结下来：
- Decoder-only结构，和GPT2等等都很相似 
- GeLU激活函数
- 2048长度的context window
- 没有偏置项（在推理搭建模型时很容易忽略的一点）
- 使用了学习的位置编码（Learned Positional Embeddings）
- 使用了BPE词表

### 项目如何setup
详细见[Galactica-README](tensorrt_llm_july-release-v1/examples/galactica/README.md#setup-envinstall-requirements-and-prepare-model)

### Setup Env(Install Requirements and Prepare model)
#### 获取docker image
```shell
docker pull registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:final_v1
docker run  -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --network host --name hackathon2023  registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:final_v1 bash
```
#### install git lfs in docker
```shell
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash 
# no need sudo in docker with root user
apt-get install git-lfs
```
#### clone this repository
```shell
git clone https://github.com/jedibobo/trt2023-final-jedibobo.git
```


#### clone model from huggingface for 125m model
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

#### Install Necessary Python Package
```shell
pip install -r requirements.txt
```

#### update tensorrt_llm package modified by the author(me)
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

### 在docker里编译运行的完整步骤
&emsp;有些部分需要科学上网，因此我这边需要**两个**命令行
#### 命令行1
```shell
cd /root/clash-linx/
./clash-linux-amd64-v3-v1.18.0 -f Config.yaml
```

#### 命令行2
```shell
docker start -i hackathon2023

export ALL_PROXY=http://127.0.0.1:7890

cd /root/workspace/trt2023-final-jedibobo/tensorrt_llm_july-release-v1/examples/galactica/

sh build_and_run_125m.sh 
#or sh build_and_run_1.3b.sh

# enable only FMHA
sh build_and_run_125m_enable_fmha.sh
# or sh build_and_run_125m_enable_fmha_weightonly.sh

# enable FMHA and weight_only
sh build_and_run_125m_enable_fmha_weightonly.sh
# or sh build_and_run_1.3b_enable_fmha_weightonly.sh
```

## 主要开发工作

### 开发工作的难点
开发中主要攻克的问题：
- 适配了Galactica模型的weight_only模式，通过修改对应的build.py和weight.py
- 搞清楚OPT和Galactica的区别，主要在于激活函数、模型参数、无BIAS结构、加载模型参数的方式。
- 参考LLAMA直接从hf读取模型的方式，编写了独立的[加载参数代码](tensorrt_llm_july-release-v1/examples/galactica/weight.py)，对BIAS全为0的模型特性加以适配。以及利用加载参数函数，编写对应的build.py中的参数加载部分。
- 参考OPT通过FT格式加载模型参数的流程，借助了其中的config.ini配置文件，成功对模型进行了正确的初始化。
- 在tensorrt_llm 目录下成功添加对应的模型文件，并可通过pip打包重新安装后利用tensorrt_llm包加载模型结构。


## 开发与优化过程
### 一切的开始
&emsp;由于本人对NLP一窍不通，在选模型时也十分迷茫，在老师们的知道下在OpenLLM的leaderboard上，通过关键词pretrained筛选，并打开几个链接后，最后选择相信Meta的开源实力，选择了Galactica模型。

&emsp;但由于对NLP的一无所知，我无法立刻通过论文或者Huggingface模型快速知道模型的结构是否新颖，也造成本工作创新性不足。
### 终于学会了用transformers加载模型
&emsp;后续根据Model Card中Huggingface加载模型的时候，打印出model的结构，发现和OPT的结构非常相似，因此我选择了利用OPT模型结构进行迁移。这里的代码在[code](tensorrt_llm_july-release-v1/examples/galactica/hf_load_inference.py)。

### 修改模型阶段(FT流程)
&emsp;通过上一步的打印，以examples中的OPT模型逐步迁移。主要有以下几个py文件：
- weight.py 模型参数加载，一般有从FT加载和从HF原模型加载两个方式
- build.py 用于创建build的配置加载模型参数，输出engine
- tensorrt_llm/models/galactica/model.py 模型结构，参考了OPT的实现

FT流程其实不太友好和直接，代码在[code](tensorrt_llm_july-release-v1/examples/galactica/hf_galactica_convert.py)。这个方式最后被证明是有问题的，我分析了保存的权重发现很多weight也是全0，造成模型输出问题，但其保存的config.ini文件中的参数是正确和必要的。
### Debug阶段-发现FT作为中转的代码中有全为0的权重，转向直接加载HF模型
&emsp;我写了个读取bin文件，然后判断里面的矩阵是否全0，以及是否有0的行或者列。发现有很多全0的矩阵，因此我认为是我写的转FT的问题，因此根据老师的建议我转向直接加载HF模型，代码在weight.py中体现。

### Debug阶段-对应模型特点的命令行参数修改
主要是两个参数：
- pre_norm
- do_layer_norm_first

&emsp;第一个是和OPT最显著的差别，也从FT输出的config.ini文件中得以体现，一次需要在build阶段在命令行指定参数。

&emsp;认识到这个参数需要修改的过程是通过对原论文的阅读，理解模型的基本结构。

### Debug阶段-发现bias被随机初始化
&emsp;在我写了一个完整的shell脚本，来作为mento方便复现工作的脚本，多次运行后的现象是随着每一次的编译模型，输出会随之改变。这里我认为是由于bias的随机初始化导致的，因此我在build.py中对bias进行了初始化，结果最终恢复正常。

### 加入Feature阶段——FMHA和weight_only
- FMHA的加入基本只需要在命令行指定参数就行
- weight_only的加入需要修改build.py和weight.py，主要是修改weight.py中的参数加载在weight_only的分支部分，以及build.py中的部分arg参数，参考llama的weightonly打开部分TRT_LLM的network配置。

### 跑实验，写报告阶段


## 优化效果
&emsp;在A10阿里云服务器里运行，使用FP16精度对于Galactica-125M和1.3B参数的两个模型在summarize任务中:
- 基础模式：分别加速**2.885**和**1.314**倍。
- 开启FMHA：且无明显精度下降的情况下，分别加速**3.166**和**1.387**倍。
- 在开启FMHA和weight_only下，分别加速**3.862**和**2.095**倍。其中1.3B模型在FMHA和weight_only下，rough score与HFmodel的差距大于1，属于有一定精度损失的情况。

具体结果可见：[Galactica-README](tensorrt_llm_july-release-v1/examples/galactica/README.md#speed-up-ratio)

速度和精度(Rouge1 Score Diff)汇总表格如下：

&emsp;在fp32、tf32和fp16精度下，galactica-125m模型在A10 GPU上的推理速度比较表格如下(batch size=1)
|  网络精度或特性   | torch_time/trt_time  | 加速比 | Rouge1 Score Diff |
| :----:| :----:|:----:|:----:|
| fp16(torch.fp16)  | 10.479/3.632  | 2.885 | 0.752 |
| fp16+FMHA(torch.fp16)  | 10.588/3.344  | 3.166 | 0.752 |
| fp16+FMHA+Weight_Only(torch.fp16)  | 10.666/2.761  | 3.862 | 0.216 |

&emsp;在fp32、tf32和fp16精度下，galactica-1.3b模型在A10 GPU上的推理速度比较表格如下(batch size=1)
|  网络精度或特性   | torch_time/trt_time  | 加速比 | Rouge1 Score Diff |
| :----:| :----:|:----:|:----:|
| fp16(torch.fp16)  | 18.968/14.435  | 1.314 | 1.024 |
| fp16+FMHA(torch.fp16)  | 19.129/13.790  | 1.387 | 0.676 |
| fp16+FMHA+Weight_Only(torch.fp16)  | 19.304/9.215  | 2.095 | **1.987** |

&emsp;在max_batch_size=8下，galactica-125m模型在A10 GPU上借助TRT_LLM的推理速度比较表格如下：

|  网络精度或特性   | bs=1/bs=8 time | 吞吐比(bs*t_1/t_bs) |
| :----:| :----:|:----:|
| fp16(torch.fp16)  | 3.515/6.496  | 4.329 | 
| fp16+FMHA(torch.fp16)  | 3.342/5.227  | 5.116 | 
| fp16+FMHA+Weight_Only(torch.fp16)  | 2.752/5.790  | 3.802 | 


&emsp;在max_batch_size=8下，galactica-1.3b模型在A10 GPU上借助TRT_LLM的推理速度比较表格如下：

|  网络精度或特性   | bs=1/bs=8 time | 吞吐比(bs*t_1/t_bs)
| :----:| :----:|:----:|
| fp16(torch.fp16)  | 14.452/34.895  | 3.313 | 
| fp16+FMHA(torch.fp16)  | 13.768/28.925  | 3.808 |
| fp16+FMHA+Weight_Only(torch.fp16)  | 9.889/32.826  | 2.410 |

&emsp;从nvidia-smi看，bs增大能显著增加GPU的利用率。



## Bug报告（可选）
无

## 送分题答案（可选）
```shell
python3 run.py --max_output_len=8
Input: Born in north-east france, Soyer trained as a
Output:  chef before moving to London in the early
```

```shell
python3 summarize.py --engine_dir=trt_engine/gpt2/fp16/1-gpu --test_hf
 --batch_size=1 --test_trt_llm --hf_model_location=gpt2 --check_accuracy --tensorrt_llm_rouge1_threshold=14

[08/17/2023-07:32:10] [TRT-LLM] [I] TensorRT-LLM (total latency: 2.6123523712158203 sec)
[08/17/2023-07:32:10] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[08/17/2023-07:32:10] [TRT-LLM] [I]   rouge1 : 21.777038933426514
[08/17/2023-07:32:10] [TRT-LLM] [I]   rouge2 : 6.1022639415477835
[08/17/2023-07:32:10] [TRT-LLM] [I]   rougeL : 16.941929893320054
[08/17/2023-07:32:10] [TRT-LLM] [I]   rougeLsum : 18.853465705869933
[08/17/2023-07:32:10] [TRT-LLM] [I] Hugging Face (total latency: 15.79952073097229 sec)
[08/17/2023-07:32:10] [TRT-LLM] [I] HF beam 0 result
[08/17/2023-07:32:10] [TRT-LLM] [I]   rouge1 : 18.182978950152904
[08/17/2023-07:32:10] [TRT-LLM] [I]   rouge2 : 5.166241888544473
[08/17/2023-07:32:10] [TRT-LLM] [I]   rougeL : 14.851620358520162
[08/17/2023-07:32:10] [TRT-LLM] [I]   rougeLsum : 16.95757748412272
```

## 经验与体会
- 比赛内容方面：这算本人第三年参加Hackathon了，每一年的难度都肉眼可见增加。但目前还和我本人代码能力的提升在一个相对平衡的状态，感觉自己也在不断进步，虽然速度不够快，但我认为是在平时工作学习没用上TRT，因此对业务流程不是那么熟悉。举个例子，上年第一的ching大佬，这次比我初赛高了不超500分（虽然人家最后3天搞定的），上年基本是我分数的5倍还要多。今年也在群里看着大佬们的讨论，尤其是Tlntin大佬的开源精神，让我深感佩服。还要感谢群里帮我解决问题的各位nv专家，帮我及时指出问题和答疑解惑，以及鼓励和帮助我提高。

  - 初赛：刚看初赛赛题还是挺懵的，首先没搞过SD，对其中step以及何处能加速都没有理解，其次没弄过这么大的模型，个人也仅仅是对CLIP还有点印象。我的经历分几个阶段吧，首先花了很多时间看懂了赛方提供的Controlnet部分的加速流程以及通过TRT的python API，对每个阶段张量的形状有更细致的了解；然后，逐个对CLIP、VAE、Control、UNET几个部分进行单独适配并替换到推理流程中；之后开始看群里的一些经验分享，比如用NV官方SD例子中的cuda graph的Engine封装、合并CLIP推理、降低Steps来踩PD12的及格线、合并Control和Unet等操作，最后两天就是不断提交刷分，以及看群里大佬分享的一些模型优化的细节，比如不打印tqdm等等赚小分的操作。真的细节决定成败。但还是有遗憾，因为对profiler使用不熟练，因此我的优化全都是看代码，而不是根据timeline有针对性地去优化。
  - 复赛：复赛内容确实有点难度，对于NLP的不熟悉以及本身不擅长通过API搭建模型的开发方式，让我在选题上花费了很多时间，最后的成果也停留在简单模型的适配上，没能在feature上更进一步。时间安排上，同时实验室方面任务比较繁杂，很难专心攻克比赛，初赛是从赛题开放2周后，复赛是选题一周，然后直到9月初第二周才开始全力弄比赛。本来都觉得要拉了，但决心再冲刺一波，毕竟选的不是很难，误打误撞发现了BIAS随机初始化的问题，感觉像是大一学C的时候不初始化变量的锅，写python多了，都忘记了这茬。

- 服务器方面：网确实有点拉，全程基本在本地服务器开发的，感觉阿里云背大锅。

- 给NV的TRT_LLM的一些建议：
- load参数的检查，我看Intro pdf的时候觉得可能在参数自动加载的时候就没这个问题了。
- 丰富下doc吧，Debug部分感觉挺重要的，也可能我太菜了，没学明白。

&emsp;遗憾和给自己的目标：
- 性能分析和Bug发现部分，感觉对我来说还有很大距离。首先Nsys用的不行，其次Debug手段还是太单一。因此，没有在Plugin上下很大功夫，不过也少了重新编译trt_llm部分库得麻烦。

