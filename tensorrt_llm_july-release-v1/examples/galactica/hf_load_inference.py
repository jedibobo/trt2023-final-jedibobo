from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import tqdm

FP16=True
INT8=False
nRound= 50

# model = "./galactica-1.3b"
model = "./galactica-125m-save"
# model = "/data/lyb/code/huggingface/galactica-6.7b/"
# model = "/data/lyb/code/huggingface/galactica-30b/"
# model = "/data/lyb/code/huggingface/llama-7B/"
# model = "../opt/opt-350m/"

if INT8:
    FP16=False

tokenizer = AutoTokenizer.from_pretrained(model)
if FP16:
    model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype=torch.float16)
elif INT8:
    model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", load_in_8bit=True)
else:
    model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")

# model_named_parameters_iter = model.named_parameters()
# model_named_parameters = dict()
# for name, param in model_named_parameters_iter:
#     if "embed" in name:
#         model_named_parameters[name] = param
#     elif "project_in" in name:
#         model_named_parameters[name] = param.permute(1, 0)
#     elif "project_out" in name:
#         model_named_parameters[name] = param
#     else:
#         model_named_parameters[name] = param.permute(1, 0) if len(param.shape) == 2 else param
model_named_parameters = dict(model.named_parameters())
print("original len:",len(model_named_parameters))
keys_to_remove = []
for key in model_named_parameters.keys():
    tensor = model_named_parameters[key]    
    # check if tensor is all zero
    if torch.allclose(tensor, torch.zeros_like(tensor)):
        print(f"{key} is all zero")
        # print(tensor)
        keys_to_remove.append(key)
for key in keys_to_remove:
    del model_named_parameters[key]  
print('del bias len',len(model_named_parameters))  
import ipdb;ipdb.set_trace()

## basic reference task
# input_text = "The Transformer architecture [START_REF]"

## summarization task
input_text = """Information overload is a major obstacle to scientific progress. The explosive growth in scientific literature and data has made it ever harder to discover useful insights in a large mass of information. Today scientific knowledge is accessed through search engines, but they are unable to organize scientific knowledge alone. In this paper we introduce Galactica: a large language model that can store, combine and reason about scientific knowledge. We train on a large scientific corpus of papers, reference material, knowledge bases and many other sources. We outperform existing models on a range of scientific tasks. On technical knowledge probes such as LaTeX equations, Galactica outperforms the latest GPT-3 by 68.2% versus 49.0%. Galactica also performs well on reasoning, outperforming Chinchilla on mathematical MMLU by 41.3% to 35.7%, and PaLM 540B on MATH with a score of 20.4% versus 8.8%. It also sets a new state-of-the-art on downstream tasks such as PubMedQA and MedMCQA dev of 77.6% and 52.9%. And despite not being trained on a general corpus, Galactica outperforms BLOOM and OPT-175B on BIG-bench. We believe these results demonstrate the potential for language models as a new interface for science. We open source the model for the benefit of the scientific community."""+"\n\nTLDR:"

input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

# run this part 100 times to get average time
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

timings = np.zeros((nRound, 1))
mem_alloc = np.zeros((nRound, 1))
mem_reserved = np.zeros((nRound, 1))

with torch.no_grad():
    for i in tqdm.tqdm(range(nRound)):
        start.record()
        outputs = model.generate(input_ids, max_length=400)
        end.record()    
        torch.cuda.synchronize()
        curr_time = start.elapsed_time(end)
        timings[i] = curr_time

        mem_alloc[i] = torch.cuda.memory_allocated()  # Memory allocated in bytes
        mem_reserved[i] = torch.cuda.memory_reserved()  # Memory reserved in bytes
        
    image_time_pytorch = timings.mean()
    avg_mem_alloc = mem_alloc.mean()
    avg_mem_reserved = mem_reserved.mean()

print("Average time for {} times of inference: {} ms".format(nRound, image_time_pytorch))
print("Average GPU memory allocated during inference: {:.2f} MB".format(avg_mem_alloc / (1024**2)))  # Convert bytes to MB
print("Average GPU memory reserved during inference: {:.2f} MB".format(avg_mem_reserved / (1024**2)))  # Convert bytes to MB
print(tokenizer.decode(outputs[0]))