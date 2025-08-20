
from utils import to_message

# from llama_cpp import Llama
# llm = Llama(
#     model_path="" ,#<path_to_model>,
#       # n_gpu_layers=-1, # Uncomment to use GPU acceleration
#       # seed=1337, # Uncomment to set a specific seed
#     n_ctx=2048, # Uncomment to increase the context window
#     chat_format="chatml",
#     verbose=False)


def translate(prompt, system_prompt) -> str:
    message = to_message(prompt, system_prompt)
    output = llm.create_chat_completion(messages=message, temperature=0)
    return output['choices'][0]['message']['content']
