# 关于复赛与TensorRT-LLM
大语言模型是计算机行业未来的重要方向，英伟达希望借助复赛的机会，加强开发团队与开发者的交流，让开发者快速上手英伟达即将正式推出的大语言模型推理加速库TensorRT-LLM，并能在未来的工作中熟练运用。

TensorRT-LLM是对TensorRT的再封装。它改善了TensorRT模型的手工搭建方式，引入了plugin提高推理性能，并加入了大量新功能。  
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
本项目主要贡献在于：使用Nvidia即将发布的TensorRT_LLM工具，对Galactica系列模型([paper](https://arxiv.org/abs/2211.09085)) ([website](https://galactica.org/)) ([official-demo](https://galactica.org/explore/)) ([huggingface-model](https://huggingface.co/facebook/galactica-6.7b))进行推理优化，具体选题内容为：用TensorRT-LLM实现新模型。

### Galactica模型简单介绍
```
Galactica models are trained on a large corpus comprising more than 360 millions in-context citations and over 50 millions of unique references normalized across a diverse set of sources. This enables Galactica to suggest citations and help discover related papers.
```
Galactica大型语言模型（LLM）正在用数百万条学术内容进行训练。它的目的是帮助研究界更好地管理"信息爆炸"。Galactica是由Meta AI与Papers with Code合作开发的。该团队认为信息过载是科学进步的一个主要障碍。"研究人员被埋没在大量的论文中，越来越无法区分有意义的和无意义的"。简单来讲，就是聚合有意义的内容，简直是我等科研狗的福音。

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

### 优化效果
在A10阿里云服务器里运行，使用FP16精度对于Galactica-125M和1.3B参数的两个模型，分别加速**2.885**和**1.314**倍。在开启FMHA，且无明显精度下降的情况下，分别加速**3.166**和**1.387**倍。在开启FMHA和weight_only下，分别加速** **和** **
具体的优化和输出结果的复现过程，也可以参照：[Galactica-README](tensorrt_llm_july-release-v1/examples/galactica/README.md)
### 在docker里编译运行的完整步骤
有些部分需要科学上网，因此我这边需要**两个** 命令行
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

sh build_and_run_125m.sh #or sh build_and_run_1.3b.sh
```

## 主要开发工作

### 开发工作的难点
开发中主要攻克的问题：
- 适配了Galactica模型的weight_only模式，以及对应的build.py和weight.py
- 搞清楚OPT和Galactica的区别，主要在于激活函数、模型参数、无BIAS结构、加载模型参数的方式。
- 参考LLAMA直接从hf读取模型的方式，编写了独立的[加载参数代码](tensorrt_llm_july-release-v1/examples/galactica/weight.py)，对BIAS全为0的模型特性加以适配。以及利用加载参数函数，编写对应的build.py中的参数加载部分。
- 参考OPT通过FT格式加载模型参数的流程，借助了其中的config.ini配置文件，成功对模型进行了正确的初始化。
- 在tensorrt_llm 目录下成功添加对应的模型文件，并可通过pip打包重新安装后利用tensorrt_llm包加载模型结构

请在这一节里总结你的工作难点与亮点。

- 如果使用 TensorRT-LLM 进行优化，描述以下方面可供选手参考：如果搭建了新模型， 请介绍模型结构有无特别之处，在模型的搭建过程中使用了什么算子，有没有通过plugin支持的新算子。如果支持新feature，请介绍这个feature具体需要修改哪些模块才能实现。如果优化已有模型，请介绍模型性能瓶颈以及解决方法。另外还可以包含工程实现以及debug过程中的难点。

## 开发与优化过程
### 一切的开始
由于本人对NLP一窍不通，在选模型时也十分迷茫，在老师们的知道下在OpenLLM的leaderboard上，通过关键词pretrained筛选，并打开几个链接后，最后选择相信Meta的开源实力，选择了Galactica模型。

但由于对NLP的一无所知，我无法立刻通过论文或者Huggingface模型快速知道模型的结构是否新颖，也造成本工作创新性不足。
### 终于学会了用transformers加载模型
后续根据Model Card中Huggingface加载模型的时候，打印出model的结构，发现和OPT的结构非常相似，因此我选择了利用OPT模型结构进行迁移。这里的代码在[code](tensorrt_llm_july-release-v1/examples/galactica/hf_load_inference.py)

### 修改模型阶段(FT流程)

### Debug阶段-发现FT作为中转的代码中有全为0的权重，转向直接加载HF模型

### Debug阶段-对应模型特点的命令行参数修改

### Debug阶段-发现bias被随机初始化
在我写了一个完整的shell脚本，来作为mento方便复现工作的脚本，多次运行后的现象是随着每一次的编译模型，输出会随之改变。这里我认为是由于bias的随机初始化导致的，因此我在build.py中对bias进行了初始化，结果最终恢复正常。


这一部分是报告的主体。请把自己假定为老师，为 TensorRT 或 TensorRT-LLM 的初学者讲述如何从原始模型出发，经过一系列开发步骤，得到优化后的 TensorRT 或 TensorRT-LLM 模型。或者你是如何一步步通过修改哪些模块添加了新feature的。

建议：

- 分步骤讲清楚开发过程
- 最好能介绍为什么需要某个特别步骤，通过这个特别步骤解决了什么问题
  - 比如，通过Nsight Systems绘制timeline做了性能分析，发现attention时间占比高且有优化空间（贴图展示分析过程），所以决定要写plugin。然后介绍plugin的设计与实现，并在timeline上显示attention这一部分的性能改进。

## 优化效果

这一部分介绍你的工作在云主机上的运行效果。如果是优化模型，需要分两部分说明：

- 精度：报告与原始模型进行精度对比测试的结果，验证精度达标。
  - 如果选用TensorRT-LLM，请跑summarize任务并使用 [Rouge](https://huggingface.co/spaces/evaluate-metric/rouge) 来对比模型优化前后的精度差距。如果精度良好，原始模型与优化模型的Rouge score的差异一般在1以内。例子见 TensorRT-LLM docker 中 /root/workspace/tensorrt_llm_july-release-v1/examples/gpt/summarize.py
  - 如果选用TensorRT，这里的精度测试指的是针对“原始模型”和“TensorRT优化模型”分别输出的数据（tensor）进行数值比较。请给出绝对误差和相对误差的统计结果（至少包括最大值、平均值与中位数）。
    - 使用训练好的权重和有意义的输入数据更有说服力。如果选手使用了随机权重和输入数据，请在这里注明。
    - 在精度损失较大的情况下，鼓励选手用训练好的权重和测试数据集对模型优化前与优化后的准确度指标做全面比较，以增强说服力。
- 性能：例如可以用图表展示不同batch size或sequence length下性能加速效果（考虑到可能模型可能比较大，可以只给batch size为1的数据）
  - 一般用原始模型作为baseline
  - 一般提供模型推理时间的加速比即可；若能提供压力测试下的吞吐提升则更好。

请注意：

- 相关测试代码也需要包含在代码仓库中，可被复现。
- 请写明云主机的软件硬件环境，方便他人参考。

## Bug报告（可选）

提交bug是对TensorRT/TensorRT-LLM的另一种贡献。发现的TensorRT/TensorRT-LLM或cookbook、或文档和教程相关bug，请提交到[github issues](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues)，并请在这里给出链接。  

对于每个bug，请标记上hackathon2023标签，并写好正文：

- 对于cookbook或文档和教程相关bug，说清楚问题即可，不必很详细。
- 对于TensorRT bug，首先确认在云主机上使用NGC docker + TensorRT 9.0.0.1可复现。
- 然后填写如下模板，并请导师复核确认（前面“评分标准”已经提到，确认有效可得附加分）：
  - Environment
    - TensorRT 9.0.0.1
    - Versions of CUDA, CUBLAS, CuDNN used
    - Container used
    - NVIDIA driver version
  - Reproduction Steps
    - Provide detailed reproduction steps for the issue here, including any commands run on the command line.
  - Expected Behavior
    - Provide a brief summary of the expected behavior of the software. Provide output files or examples if possible.
  - Actual Behavior
    - Describe the actual behavior of the software and how it deviates from the expected behavior. Provide output files or examples if possible.
  - Additional Notes
    - Provide any additional context here you think might be useful for the TensorRT team to help debug this issue (such as experiments done, potential things to investigate).

## 送分题答案（可选）

如果你做了送分题，请把答案写在这里。

## 经验与体会（可选）

欢迎在这里总结经验，抒发感慨。
