import configparser
import time
from pathlib import Path

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm.models import GALAILMHeadModel
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy

def extract_layer_idx(name):
    ss = name.split('.')
    for s in ss:
        if s.isdigit():
            return s
    return None

def check_zero_rows_columns(arr, identifier="Array"):
    """
    Checks if the input array contains any zero rows or columns.
    
    Parameters:
    arr (numpy.ndarray): The input array to check.
    identifier (str): A string identifier for logging purposes.
    
    Returns:
    None
    """
    if len(arr.shape) == 1:
        zero_rows = (arr.sum() == 0)
        zero_columns = False
    elif len(arr.shape) == 2:
        zero_rows = (arr.sum(axis=1) == 0).any()
        zero_columns = (arr.sum(axis=0) == 0).any()
    else:
        raise ValueError(f"{identifier} dimensions should be 1 or 2")

    if zero_rows:
        print(f"{identifier} contains at least one zero row.")
    if zero_columns:
        print(f"{identifier} contains at least one zero column.")

def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx])
    elif len(v.shape) == 2:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])
    return None

def load_from_hf_galai(tensorrt_llm_galai,
                       hf_galai,
                       rank=0,
                       tensor_parallel=1,
                       dtype="float32",
                       multi_query_mode=False):
    tensorrt_llm.logger.info('Loading weights from HF LLaMA...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_galai, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()

    model_params = dict(hf_galai.named_parameters())
    # remove zero biases 
    keys_to_remove = []
    for key in model_params.keys():
        tensor = model_params[key]    
        # check if tensor is all zero
        if torch.allclose(tensor, torch.zeros_like(tensor)):
            print(f"{key} is all zero")
            # print(tensor)
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del model_params[key]
    for l in range(hf_galai.config.num_hidden_layers):
        prefix = f'model.decoder.layers.{l}.self_attn.'
        q_weight = model_params[prefix + 'q_proj.weight']
        k_weight = model_params[prefix + 'k_proj.weight']
        v_weight = model_params[prefix + 'v_proj.weight']
        if multi_query_mode:
            head_size = tensorrt_llm_galai.hidden_size // tensorrt_llm_galai.num_heads
            assert k_weight.shape[0] == tensor_parallel * head_size
            assert v_weight.shape[0] == tensor_parallel * head_size
            qkv_weight = [q_weight, k_weight, v_weight]
        else:
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

        model_params[prefix + 'qkv_proj.weight'] = qkv_weight

    torch_dtype = str_dtype_to_torch(dtype)

    for k, v in model_params.items():
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(torch_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(torch_dtype).detach().cpu())
        if 'model.decoder.embed_tokens.weight' in k:
            check_zero_rows_columns(v, identifier=k)
            tensorrt_llm_galai.embedding.vocab_embedding.weight.value = v
            tensorrt_llm_galai.lm_head.weight.value = np.ascontiguousarray(
                split(v, tensor_parallel, rank))
        elif 'model.decoder.embed_positions.weight' in k:
            check_zero_rows_columns(v[2:], identifier=k)
            tensorrt_llm_galai.embedding.position_embedding.weight.value = v[2:]
        elif 'model.decoder.final_layer_norm.weight' in k:
            check_zero_rows_columns(v, identifier=k)
            tensorrt_llm_galai.ln_f.weight.value = v
        # elif 'lm_head.weight' in k:
        #     tensorrt_llm_galai.lm_head.weight.value = np.ascontiguousarray(
        #         split(v, tensor_parallel, rank))
        else:
            check_zero_rows_columns(v, identifier=k)
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            idx = int(layer_idx)
            if idx >= tensorrt_llm_galai._num_layers:
                continue
            if 'final_layer_norm.weight' in k:
                tensorrt_llm_galai.layers[idx].post_layernorm.weight.value = v
            elif 'self_attn_layer_norm.weight' in k:
                dst = tensorrt_llm_galai.layers[idx].input_layernorm.weight
                dst.value = v
            elif 'self_attn.qkv_proj.weight' in k:
                dst = tensorrt_llm_galai.layers[idx].attention.qkv.weight
                if multi_query_mode:
                    assert isinstance(v, list) and len(v) == 3
                    wq = split(v[0], tensor_parallel, rank)
                    wk = split(v[1], tensor_parallel, rank)
                    wv = split(v[2], tensor_parallel, rank)
                    split_v = np.concatenate((wq, wk, wv))
                else:
                    q_emb = v.shape[0] // 3
                    model_emb = v.shape[1]
                    v = v.reshape(3, q_emb, model_emb)
                    split_v = split(v, tensor_parallel, rank, dim=1)
                    split_v = split_v.reshape(3 * (q_emb // tensor_parallel),
                                              model_emb)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_galai.layers[
                        idx].attention.qkv.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'self_attn.out_proj.weight' in k:
                dst = tensorrt_llm_galai.layers[idx].attention.dense.weight
                split_v = split(v, tensor_parallel, rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_galai.layers[
                        idx].attention.dense.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'fc1.weight' in k:
                dst = tensorrt_llm_galai.layers[idx].mlp.fc.weight
                split_v = split(v, tensor_parallel, rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_galai.layers[
                        idx].mlp.gate.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'fc2.weight' in k:
                dst = tensorrt_llm_galai.layers[idx].mlp.proj.weight
                split_v = split(v, tensor_parallel, rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_galai.layers[
                        idx].mlp.proj.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    return

def parse_ft_config(ini_file):
    gpt_config = configparser.ConfigParser()
    gpt_config.read(ini_file)

    n_embd = gpt_config.getint('gpt', 'n_embd')
    n_head = gpt_config.getint('gpt', 'n_head')
    n_layer = gpt_config.getint('gpt', 'n_layer')
    n_positions = gpt_config.getint('gpt', 'n_positions')
    vocab_size = gpt_config.getint('gpt', 'vocab_size')
    do_layer_norm_before = gpt_config.getboolean('gpt',
                                                 'do_layer_norm_before',
                                                 fallback=True)

    return n_embd, n_head, n_layer, n_positions, vocab_size, do_layer_norm_before


def load_from_ft(tensorrt_llm_gpt: GALAILMHeadModel,
                 dir_path,
                 rank=0,
                 tensor_parallel=1,
                 fp16=False):
    tensorrt_llm.logger.info('Loading weights from FT...')
    tik = time.perf_counter()

    n_embd, n_head, n_layer, n_positions, vocab_size, do_layer_norm_before = parse_ft_config(
        Path(dir_path) / 'config.ini')
    np_dtype = np.float16 if fp16 else np.float32

    def fromfile(dir_path, name, shape=None):
        p = dir_path + '/' + name
        if Path(p).exists():
            t = np.fromfile(p, dtype=np_dtype)
            if shape is not None:
                t = t.reshape(shape)
            return t
        return None

    pe = fromfile(dir_path, 'model.wpe.bin', [n_positions, n_embd])
    if pe is not None:
        tensorrt_llm_gpt.embedding.position_embedding.weight.value = (pe)
    tensorrt_llm_gpt.embedding.vocab_embedding.weight.value = (fromfile(
        dir_path, 'model.wte.bin', [vocab_size, n_embd]))
    if do_layer_norm_before:
        tensorrt_llm_gpt.ln_f.bias.value = (fromfile(
            dir_path, 'model.final_layernorm.bias.bin'))
        tensorrt_llm_gpt.ln_f.weight.value = (fromfile(
            dir_path, 'model.final_layernorm.weight.bin'))
    # share input embedding
    lm_head_weight = fromfile(dir_path, 'model.lm_head.weight.bin',
                              [vocab_size, n_embd])
    if lm_head_weight is None:
        lm_head_weight = fromfile(dir_path, 'model.wte.bin',
                                  [vocab_size, n_embd])
    if vocab_size % tensor_parallel != 0:
        # padding
        vocab_size_padded = tensorrt_llm_gpt.lm_head.out_features * tensor_parallel
        pad_width = vocab_size_padded - vocab_size
        lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)),
                                'constant',
                                constant_values=0)
    tensorrt_llm_gpt.lm_head.weight.value = np.ascontiguousarray(
        split(lm_head_weight, tensor_parallel, rank))
    for i in range(n_layer):
        tensorrt_llm_gpt.layers[i].input_layernorm.weight.value = (fromfile(
            dir_path, 'model.layers.' + str(i) + '.input_layernorm.weight.bin'))
        tensorrt_llm_gpt.layers[i].input_layernorm.bias.value = (fromfile(
            dir_path, 'model.layers.' + str(i) + '.input_layernorm.bias.bin'))
        t = fromfile(
            dir_path, 'model.layers.' + str(i) +
            '.attention.query_key_value.weight.' + str(rank) + '.bin',
            [n_embd, 3 * n_embd // tensor_parallel])
        if t is not None:
            dst = tensorrt_llm_gpt.layers[i].attention.qkv.weight
            dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
        t = fromfile(
            dir_path, 'model.layers.' + str(i) +
            '.attention.query_key_value.bias.' + str(rank) + '.bin')
        if t is not None:
            dst = tensorrt_llm_gpt.layers[i].attention.qkv.bias
            dst.value = np.ascontiguousarray(t)

        dst = tensorrt_llm_gpt.layers[i].attention.dense.weight
        t = fromfile(
            dir_path, 'model.layers.' + str(i) + '.attention.dense.weight.' +
            str(rank) + '.bin', [n_embd // tensor_parallel, n_embd])
        dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

        dst = tensorrt_llm_gpt.layers[i].attention.dense.bias
        dst.value = fromfile(
            dir_path, 'model.layers.' + str(i) + '.attention.dense.bias.bin')

        dst = tensorrt_llm_gpt.layers[i].post_layernorm.weight
        dst.value = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.post_attention_layernorm.weight.bin')

        dst = tensorrt_llm_gpt.layers[i].post_layernorm.bias
        dst.value = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.post_attention_layernorm.bias.bin')
        t = fromfile(
            dir_path, 'model.layers.' + str(i) + '.mlp.dense_h_to_4h.weight.' +
            str(rank) + '.bin', [n_embd, 4 * n_embd // tensor_parallel])
        tensorrt_llm_gpt.layers[i].mlp.fc.weight.value = np.ascontiguousarray(
            np.transpose(t, [1, 0]))
        tensorrt_llm_gpt.layers[i].mlp.fc.bias.value = fromfile(
            dir_path, 'model.layers.' + str(i) + '.mlp.dense_h_to_4h.bias.' +
            str(rank) + '.bin')
        t = fromfile(
            dir_path, 'model.layers.' + str(i) + '.mlp.dense_4h_to_h.weight.' +
            str(rank) + '.bin', [4 * n_embd // tensor_parallel, n_embd])
        tensorrt_llm_gpt.layers[i].mlp.proj.weight.value = (
            np.ascontiguousarray(np.transpose(t, [1, 0])))
        tensorrt_llm_gpt.layers[i].mlp.proj.bias.value = fromfile(
            dir_path, 'model.layers.' + str(i) + '.mlp.dense_4h_to_h.bias.bin')

    tok = time.perf_counter()
    # t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    # tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    elapsed_time = tok - tik
    # 格式化时间间隔，包括秒的小数部分
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    time_format = f"{int(hours):02d}:{int(minutes):02d}:{seconds:.6f}"
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {time_format}')
