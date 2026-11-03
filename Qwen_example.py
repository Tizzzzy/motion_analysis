import argparse
from typing import Dict, Iterable, List
from vllm import LLM, SamplingParams

# --- Configuration & Paths ---
MODEL_PATH = "/projects/p32143/cache/qwen38_27b" # Path to your Qwen model
MODEL_ID_NAME = "Qwen/Qwen3.8-27B" # Used for saving the output folder


def batched(items: List, batch_size: int) -> Iterable[List]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def format_with_chat_template(llm: LLM, content: list, enable_thinking: bool = False) -> str:
    tokenizer = llm.get_tokenizer()
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Some tokenizer/vLLM versions pass Qwen chat-template kwargs this way.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max_model_len", type=int, default=262144)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--enforce_eager", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    ap.add_argument("--disable_log_stats", action="store_true", default=True)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0)
    ap.add_argument("--top_p", type=float, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--use_chat_template", action="store_true", default=True)
    ap.add_argument("--enable_thinking", action="store_true", help="Enable Qwen thinking mode in chat template.")
    args = ap.parse_args()

    print("🚀 Initializing vLLM Engine (This happens only once)...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True, # Hardcoded to True in your original script
        trust_remote_code=args.trust_remote_code,
        disable_log_stats=args.disable_log_stats,
        generation_config="vllm",
        # Note: limit_mm_per_prompt is commented out. Only use it if you are 
        # using a Vision model (like Qwen-VL). Pure text models will throw an error.
        # limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0}, 
    )

    sampling_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if args.seed is not None:
        sampling_kwargs["seed"] = args.seed
    sampling = SamplingParams(**sampling_kwargs)

    # --- TEXT PROMPT ---
    prompt_text = "Who is the president of the United States in 2023? Please answer in one sentence."
                
    # FIX 1: Initialize the content list before appending to it
    content = []
    
    # Note: Text models usually expect `content` as a string, but the 
    # [{"type": "text", "text": "..."}] dict format is accepted by modern Qwen tokenizers
    content.append({"type": "text", "text": prompt_text})

    # Format the prompt string via the tokenizer
    if args.use_chat_template:
        final_prompt_str = format_with_chat_template(llm, content, args.enable_thinking)
    else:
        final_prompt_str = prompt_text

    # FIX 2: Initialize the batch array before appending to it
    batch_vllm_inputs = []
    
    # --- PREPARE vLLM DICTIONARY INPUT ---
    vllm_input = {"prompt": final_prompt_str}
    batch_vllm_inputs.append(vllm_input)

    print("🧠 Generating response...")
    # Generate outputs using vLLM engine
    try:
        outputs = llm.generate(batch_vllm_inputs, sampling)
        
        for idx, out in enumerate(outputs):
            predicted_text = out.outputs[0].text.strip() if out.outputs else ""
            
            # FIX 3: Actually print the result to the console
            print(f"\n--- Output {idx + 1} ---")
            print(predicted_text)
            print("-" * 20)

    except Exception as e:
        print(f"❌ Error during batch generation: {e}")
        
if __name__ == "__main__":
    main()