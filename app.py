import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
import time
import numpy as np
from torch.nn import functional as F
import os
from threading import Thread


print(f"Starting to load the model to memory")


USER_TOKEN = '<|USER|>'
ASSISTANT_TOKEN = '<|ASSISTANT|>'

language_model = AutoModelForCausalLM.from_pretrained(
    "stabilityai/stablelm-tuned-alpha-7b",
    load_in_8bit=True,
    device_map='auto',
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(
    "stabilityai/stablelm-tuned-alpha-7b"
)


print(f"Sucessfully loaded the model to the memory")

start_message = """<|SYSTEM|># StableAssistant
- StableAssistant is A helpful and harmless Open Source AI Language Model developed by Stability and CarperAI.
- StableAssistant is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableAssistant is more than just an information source, StableAssistant is also able to write poetry, short stories, and make jokes.
- StableAssistant will refuse to participate in anything that could harm a human."""


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0] # TODO: Verify what tokens are this
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


def chat(curr_system_message, history):
    # Initialize a StopOnTokens object
    stop = StopOnTokens()

    # Construct the input message string for the model by concatenating the current system message and conversation history
    messages = curr_system_message + \
        "".join(["".join([USER_TOKEN+item[0], ASSISTANT_TOKEN+item[1]]) for item in history])

    # Tokenize the messages string
    model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, timeout=30., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=1040,
        do_sample=True,
        top_p=0.95,
        top_k=1000,
        temperature=1.0,
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
    gr.Markdown("## StableLM-Tuned-Alpha-7b Chat")
    chatbot = gr.Chatbot().style(height=700)
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
    
    system_msg = gr.Textbox(
        start_message,
        label="System Message",
        interactive=False,
        visible=True
    )

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
