import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from huggingface_hub import HfFolder
import time
import numpy as np
from torch.nn import functional as F
import os
from threading import Thread

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    TaskType,
    PeftConfig,
    PeftModel
)


print(f"Starting to load the model to memory")


USER_TOKEN = '<|USER|>'
ASSISTANT_TOKEN = '<|ASSISTANT|>'

HfFolder.save_token(os.getenv("HF_TOKEN"))

peft_model_id = "rjac/temp_model"
config = PeftConfig.from_pretrained(peft_model_id)
# Custom One
tokenizer = AutoTokenizer.from_pretrained(peft_model_id,use_auth_token=True)
# tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
language_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    load_in_8bit=True,
    device_map='auto',
    torch_dtype=torch.float16,
)

language_model = PeftModel.from_pretrained(language_model, peft_model_id)

print(f"Sucessfully loaded the model to the memory")

start_message = """<|SYSTEM|># Nutrition Assistant: Answer questions related to Senza nutrition app, using the context provided within triple backticks.\nIf a question is unrelated to the app, respond: I am sorry, I\'m afraid I cannot answer that question.\n```\n{}\n```\n"""

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        #stop_ids = [50278, 50279, 50277, 1, 0] # TODO: Verify what tokens are this
        stop_ids = [535, 50277, 50278, 50279, 1, 0]
        #stop_ids = [1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def user(message, history):
    # Append the user's message to the conversation history
    return "", history[-1:] + [[message, ""]]
    # return "", [[message, ""]]


def chat(curr_system_message, history):
    # Initialize a StopOnTokens object
    stop = StopOnTokens()

    system_message = start_message.format(curr_system_message)
    # Construct the input message string for the model by concatenating the current system message and conversation history
    messages = system_message + \
        "".join(["".join([USER_TOKEN+item[0], ASSISTANT_TOKEN+item[1]]) for item in history])
    print(messages)
    # Tokenize the messages string
    model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, timeout=30., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=150,
        do_sample=True,
        top_p=0.90,
        top_k=1000,
        temperature=0.7,
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


with gr.Blocks() as demo:
    # history = gr.State([])
    gr.Markdown("## StableLM-Senza-alpha-7b Chat")

    system_msg = gr.Textbox(
        "User Info",
        label="System Message",
        interactive=True,
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
                submit = gr.Button("Submit")
                stop = gr.Button("Stop")
                clear = gr.Button("Clear")

    chatbot = gr.Chatbot().style(height=250)


    submit_event = msg.submit(
        fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False
    ).then(
        fn=chat, inputs=[system_msg, chatbot], outputs=[chatbot], queue=True
    )

    submit_click_event = submit.click(
        fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False
    ).then(
        fn=chat, inputs=[system_msg, chatbot], outputs=[chatbot], queue=True
    )
    
    stop.click(
        fn=None, inputs=None, outputs=None, cancels=[submit_event, submit_click_event], queue=False
    )
    
    clear.click(lambda: None, None, [chatbot], queue=False)

demo.queue(max_size=32, concurrency_count=6)
demo.launch(server_name="0.0.0.0")
