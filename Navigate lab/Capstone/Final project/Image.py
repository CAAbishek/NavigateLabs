import torch
import time
import tracemalloc
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, BertTokenizer, GenerationConfig

def load_image_model():
    model_id = "IAMJB/chexpert-mimic-cxr-findings-baseline"
    
    model = VisionEncoderDecoderModel.from_pretrained(model_id).eval()
    tokenizer = BertTokenizer.from_pretrained(model_id)
    processor = ViTImageProcessor.from_pretrained(model_id)

    generation_args = {
        "bos_token_id": model.config.bos_token_id,
        "eos_token_id": model.config.eos_token_id,
        "pad_token_id": model.config.pad_token_id,
        "num_return_sequences": 1,
        "max_length": 128,
        "use_cache": True,
        "beam_width": 2,
        "decoder_start_token_id": tokenizer.cls_token_id,
    }

    return model, tokenizer, processor, generation_args

def generate_caption_with_latency(image, model, tokenizer, processor, generation_args):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    
    with torch.no_grad():
        tracemalloc.start()
        start_time = time.time()

        generated_ids = model.generate(
            pixel_values,
            generation_config=GenerationConfig(**generation_args)
        )

        end_time = time.time()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    latency = end_time - start_time
    throughput = 1 / latency
    memory = peak_mem / (1024 ** 2)  # Convert bytes to MB

    return text, latency, throughput, memory
