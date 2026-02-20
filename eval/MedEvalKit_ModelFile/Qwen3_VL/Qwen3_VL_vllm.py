from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import os
import re

class Qwen3_VL:
    def __init__(self,model_path,args):
        super().__init__()
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=int(os.environ.get("tensor_parallel_size", 1)),
            trust_remote_code=True,
            enforce_eager=True,
            gpu_memory_utilization=0.9,
            enable_prefix_caching=True,
            limit_mm_per_prompt = {"image": args.max_image_num},
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens= args.max_new_tokens,
            stop_token_ids=[],
        )

    def process_messages(self,messages):
        current_messages = []

        if "messages" in messages:
            messages = messages["messages"]
            for message in messages:
                role = message["role"]
                content = message["content"]
                current_messages.append({"role":role,"content":[{"type":"text","text":content}]})

        else:
            prompt = messages["prompt"]
            if "system" in messages:
                system_prompt = messages["system"]
                current_messages.append({"role":"system","content":system_prompt})
            if "image" in messages:
                image = messages["image"]
                current_messages.append({"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]})
            elif "images" in messages:
                content = []
                for i,image in enumerate(messages["images"]):
                    content.append({"type":"text","text":f"<image_{i+1}>: "})
                    content.append({"type":"image","image":image})
                content.append({"type":"text","text":messages["prompt"]})
                current_messages.append({"role":"user","content":content})
            else:
                current_messages.append({"role":"user","content":[{"type":"text","text":prompt}]})

        messages = current_messages
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs

        llm_inputs = {
            "prompt": prompt
        }
        if mm_data:
            llm_inputs["multi_modal_data"] = mm_data
        return llm_inputs

    def _strip_think_tokens(self, text):
        """Remove <think>...</think> or <thinking>...</thinking> blocks from model output."""
        if '</think>' in text:
            text = text.split('</think>')[-1].strip()
        elif '</thinking>' in text:
            text = text.split('</thinking>')[-1].strip()
        return text

    def generate_output(self,messages):
        llm_inputs = self.process_messages(messages)
        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        generated_text = outputs[0].outputs[0].text
        return self._strip_think_tokens(generated_text)

    def generate_outputs(self,messages_list):
        llm_inputs_list = [self.process_messages(messages) for messages in messages_list]
        outputs = self.llm.generate(llm_inputs_list, sampling_params=self.sampling_params)
        res = []
        for output in outputs:
            generated_text = output.outputs[0].text
            res.append(self._strip_think_tokens(generated_text))
        return res
