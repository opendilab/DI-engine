# from transformers import AutoTokenizer, RwkvForCausalLM, AutoModelForSeq2SeqLM
# import torch
# import torch.functional as F
# import numpy as np
#
# tokenizer = AutoTokenizer.from_pretrained("/mnt/nfs/whl/rwkv-7b", trust_remote_code=True)
# model = RwkvForCausalLM.from_pretrained("/mnt/nfs/whl/rwkv-7b", trust_remote_code=True)
# model = model.half().cuda()
# model.eval()
#
#
# def sample_logits(out, temperature=1.0, top_p=0.8):
#     probs = torch.softmax(out, dim=-1).cpu().numpy()
#     sorted_probs = np.sort(probs)[::-1]
#     cumulative_probs = np.cumsum(sorted_probs)
#     cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
#     probs[probs < cutoff] = 0
#     if temperature != 1.0:
#         probs = probs.pow(1.0 / temperature)
#     probs = probs / np.sum(probs)
#     out = np.random.choice(a=len(probs), p=probs)
#     return out
#
#
# # Inference
# str_inputs = "My name is Sam, I am a senior at"
# inputs = tokenizer(str_inputs, return_tensors="pt").to('cuda')
# outputs = model(**inputs, labels=inputs["input_ids"])
# out, state = outputs.logits, outputs.state
#
# with torch.no_grad():
#     for TRIAL in range(1):
#         for i in range(20):
#             token = sample_logits(out[0, -1])
#             tmp = tokenizer.decode([token])
#             str_inputs = str_inputs + tmp
#             print(str_inputs)
#             inputs = tokenizer(str_inputs, return_tensors="pt").to('cuda')
#             outputs = model(**inputs, labels=inputs["input_ids"])
#             out, state = outputs.logits, outputs.state
