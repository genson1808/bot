import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_community.llms import LlamaCpp


model_id = "meta-llama/Llama-2-7b-hf"

# Enable MPS check
if torch.backends.mps.is_available():
    print("MPS backend is available.")
else:
    print("MPS backend is not available, falling back to CPU.")

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16  # Lower memory precision
)



def get_hf_llm(model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
                max_new_token = 1024,
                **kwargs):
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     # quantization_config=nf4_config,
    #     trust_remote_code=True,
    #     torch_dtype=torch.bfloat16,  # Set lower precision to save memory
    #     # device_map="auto"  # Automatically distribute model on available devices
    # )
    # model.to("mps")

    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # # device = 0 if torch.cuda.is_available() else -1

    # model_pipeline = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     max_new_tokens=max_new_token,
    #     pad_token_id=tokenizer.eos_token_id,
    #     device_map="auto",
    #     # device="mps"
    # )

    # llm = HuggingFacePipeline(
    #     pipeline=model_pipeline,
    #     model_kwargs=kwargs,
    # )

    # Callbacks support token-wise streaming
    # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        # model_path="/Users/genson1808/workspace/ai/chatbot/models/phi-2.Q5_K_M.gguf",
        model_path="/Users/genson1808/workspace/ai/chatbot/models/vinallama-7b-chat_q5_0.gguf",
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        # callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
        # temperature=0.01,
        n_ctx=max_new_token,
        model_kwargs=kwargs,
    )
    return llm