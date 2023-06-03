import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)
from huggingface_hub import HfFolder
import time
import numpy as np
from torch.nn import functional as F
import os
from threading import Thread
from context_builder import build_user_context

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    TaskType,
    PeftConfig,
    PeftModel
)


print(f"Starting to load the model to memory")

#this can be setup from the Dialog template
USER_TOKEN = '>>QUESTION<<\n'
ASSISTANT_TOKEN = '>>ANSWER<<\n'
EOS_TOKEN = '<|endoftext|>\n'

HfFolder.save_token(os.getenv("HF_TOKEN"))
#peft_model_id = "rjac/temp_modelv3"
#peft_model_id = "rjac/senza-chat-stablelm-2-0"
base_model_name_or_path = "rjac/senza-chat-falcon7b-v0"

#config = PeftConfig.from_pretrained(peft_model_id)
# Custom One
tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path,use_auth_token=True)
tokenizer.eos_token_id = 11
tokenizer.pad_token_id = 39

# tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
#language_model = AutoModelForCausalLM.from_pretrained(
#    config.base_model_name_or_path,
#    load_in_8bit=True,
#    device_map='auto',
#    torch_dtype=torch.float16,
#)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    #bnb_4bit_compute_dtype=torch.bfloat16,
)

#model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map={"":0})
language_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, quantization_config=bnb_config, trust_remote_code=True, device_map="auto")

# if using PEFT QLORA
#language_model = PeftModel.from_pretrained(language_model, peft_model_id)

print(f"Sucessfully loaded the model to the memory")
#this can be setup from the Dialog template
#start_message = """<|system|>\n# Nutrition Assistant: Answer questions related to Senza nutrition app, using the context provided within triple backticks.\nIf a question is unrelated to the app, respond: I am sorry, I\'m afraid I cannot answer that question.\n```\n{}\n```\n<|end|>\n"""
start_message = """>>INTRODUCTION<<\n# Coach: Answer users questions related to the Senza nutrition app using the context provided between triple backticks. \nIf a question is unrelated to the app or nutrition, respond: I am sorry, I\'m afraid I cannot answer that question.\n```\n{}the net carb content of food is calculated by taking the grams of total carbs and subtracting grams of fiber and sugar alcohols.\n```\n<|endoftext|>\n"""


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        #stop_ids = [50278, 50279, 50277, 1, 0] # TODO: Verify what tokens are this
        stop_ids = [11, 39, 2, 6, 5]
        #stop_ids = [1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def user(message, history):
    # Append the user's message to the conversation history
    # return "", history + [[message, ""]]
    return "", history[-2:] + [[message, ""]]
    # return "", [[message, ""]]


def chat(curr_system_message, history, temperature):
    # Initialize a StopOnTokens object
    stop = StopOnTokens()

    system_message = start_message.format(curr_system_message)
    # Construct the input message string for the model by concatenating the current system message and conversation history
    messages = system_message + \
        "".join(["".join([USER_TOKEN+item[0]+EOS_TOKEN, ASSISTANT_TOKEN+item[1]+EOS_TOKEN]) for item in history])
    
    messages = messages[:-len(EOS_TOKEN)]
    print(messages)
    print(f"temperature: {temperature}")
    # Tokenize the messages string
    model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
    token_type_ids = model_inputs.pop("token_type_ids")
    streamer = TextIteratorStreamer(tokenizer, timeout=30., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.90,
        top_k=1000,
        temperature=temperature,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
    )
    t = Thread(target=language_model.generate, kwargs=generate_kwargs)
    t.start()

    # print(history)
    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        # print(new_text)
        partial_text += new_text
        history[-1][1] = partial_text
        # Yield an empty string to cleanup the message textbox and the updated conversation history
        yield history
    return partial_text


def user_context(user_id):
    user_info = build_user_context(user_id)
    return user_info

with gr.Blocks() as demo:
    # history = gr.State([])
    gr.Markdown("## Senza AI Coach - Falcon-7B Version 0")

    with gr.Row():

        user_id_holder = gr.Textbox(
            label="User ID",
            placeholder="####",
            show_label=True
        ).style(container=False)        

        search_user_info = gr.Button("Search")

    system_msg = gr.Textbox(
        "User Info",
        label="System Message",
        interactive=False,
        visible=True
    )


    with gr.Row():
        with gr.Column():            
            msg = gr.Textbox(
                label="Chat Message Box",
                placeholder="Chat Message Box",
                show_label=False
            ).style(container=False)

        
        with gr.Column():
            with gr.Row():
                temperature = gr.Slider( minimum=0.05, maximum=3.0, value=1.0, step=0.05, interactive=True, label="Temperature")
                submit = gr.Button("Submit")
                stop = gr.Button("Stop")
                clear = gr.Button("Clear")
        

    chatbot = gr.Chatbot().style(height=350)


    search_user_info_event = search_user_info.click(fn=user_context, inputs=[user_id_holder],outputs=[system_msg], queue=False)


    submit_event = msg.submit(
        fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False
    ).then(
        fn=chat, inputs=[system_msg, chatbot,temperature], outputs=[chatbot], queue=True
    )

    submit_click_event = submit.click(
        fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False
    ).then(
        fn=chat, inputs=[system_msg, chatbot, temperature], outputs=[chatbot], queue=True
    )
    
    stop.click(
        fn=None, inputs=None, outputs=None, cancels=[submit_event, submit_click_event], queue=False
    )
    
    clear.click(lambda: None, None, [chatbot], queue=False)

demo.queue(max_size=32, concurrency_count=3)
demo.launch(server_name="0.0.0.0")
