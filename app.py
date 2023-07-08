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
from transformers.utils import is_torch_bf16_gpu_available
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

#model_id = "rjac/falcon7B-recsys-senza"
model_id = "senza-recsys-stablelm-alpha-7B-runpod"
peft_config = PeftConfig.from_pretrained(model_id)
model_name = peft_config.base_model_name_or_path

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,

)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False
language_model = PeftModel.from_pretrained(model, model_id).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

eos = tokenizer.encode("<|endoftext|>###")

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [11, 19468, 39, 2, 6, 5]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def user(message, history):
    return "", [[message, ""]]


def chat(curr_system_message, history, temperature):
    # Initialize a StopOnTokens object
    stop = StopOnTokens()
    history = [["", ""]]
    messages = curr_system_message
    #messages = messages[:-len(EOS_TOKEN)]

    print(messages)
    print(f"temperature: {temperature}")
    # Tokenize the messages string
    model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
    token_type_ids = model_inputs.pop("token_type_ids")
    streamer = TextIteratorStreamer(tokenizer, timeout=60., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=500,
        do_sample=True,
        top_p=0.90,
        top_k=10,
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


def user_context(user_id, meal_time_diner = "Dinner"):
    user_info = build_user_context(user_id,meal_time_diner)
    return user_info

with gr.Blocks() as demo:
    # history = gr.State([])
    gr.Markdown(f"## Senza AI RecSys - model id: {model_id}")

    with gr.Row():

        user_id_holder = gr.Textbox(
            label="User ID",
            placeholder="####",
            show_label=True
        ).style(container=False)  

        meal_time_holder = gr.Textbox(
            label="User ID",
            placeholder="Breakfast | Lunch | Dinner",
            show_label=True
        ).style(container=False)  

        search_user_info = gr.Button("Search")

    system_msg = gr.Textbox(
        "User Info",
        label="System Message",
        interactive=True,
        visible=True
    ).style(height=1250)

    with gr.Row():
        #with gr.Column():            
            #msg = gr.Textbox(
            #    label="Chat Message Box",
            #    placeholder="Chat Message Box",
            #    show_label=False
            #).style(container=False)
        with gr.Column():
            with gr.Row():
                temperature = gr.Slider( minimum=0.05, maximum=3.0, value=1.0, step=0.05, interactive=True, label="Temperature")
                submit = gr.Button("Submit")
                stop = gr.Button("Stop")
                clear = gr.Button("Clear")
        
    chatbot = gr.Chatbot().style(height=350)


    search_user_info_event = search_user_info.click(fn=user_context, inputs=[user_id_holder, meal_time_holder],outputs=[system_msg], queue=False)
    #submit_event = msg.submit(
    #    fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False
    #).then(
    #    fn=chat, inputs=[system_msg, chatbot, temperature], outputs=[chatbot], queue=True
    #)

    #submit_click_event = submit.click(
    #    fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False
    #).then(
    #    fn=chat, inputs=[system_msg, chatbot, temperature], outputs=[chatbot], queue=True
    #)

    submit_click_event = submit.click(
        fn=chat, inputs=[system_msg, chatbot, temperature], outputs=[chatbot], queue=True
    )


    stop.click(
        fn=None, inputs=None, outputs=None, cancels=[submit_click_event], queue=False
    )
    
    clear.click(lambda: None, None, [chatbot], queue=False)

print(f"BF16 support is {is_torch_bf16_gpu_available()}")
demo.queue(max_size=32, concurrency_count=3)
demo.launch(server_name="0.0.0.0")
