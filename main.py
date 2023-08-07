# -*- coding: utf-8 -*-
# Author: 薄荷你玩
# Date: 2023/07/24

from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "D:\code\Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().cuda()

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids.to('cuda'), max_length=300)
response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(response)

# # 最多携带历史会话轮数
# max_history = 5
#
while True:
    inputs = tokenizer(input(">> "), return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids.to('cuda'), max_length=300)
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(response)
