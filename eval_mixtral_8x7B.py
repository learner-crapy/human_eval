# mistral 7B
from transformers import (
    PreTrainedModel,
    MistralPreTrainedModel,
    PreTrainedTokenizer,
    MistralForCausalLM,
    AutoTokenizer
)

from core import filter_code, run_eval, fix_indents
import os
import torch

TOKEN = ""


@torch.inference_mode()
def generate_batch_completion(
        model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt, batch_size
) -> list[str]:
    input_batch = [prompt for _ in range(batch_size)]
    inputs = tokenizer(input_batch, return_tensors='pt').to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True
    )

    return [filter_code(fix_indents(completion)) for completion in batch_completions]


if __name__ == "__main__":
    num_samples_per_task = 20
    out_path = "results/mistral7B/eval.jsonl"
    os.makedirs('results/mistral7B', exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained('/home/ludaze/Docker/Llama/Mixtral/Mistral-7B-v0.1')
    model = torch.compile(
        MistralForCausalLM.from_pretrained(
            pretrained_model_name_or_path='/home/ludaze/Docker/Llama/Mixtral/Mistral-7B-v0.1',
            torch_dtype=torch.bfloat16
        ).eval()
        .to('cuda')
    )

    net = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    run_eval(
        model,
        tokenizer,
        num_samples_per_task,
        out_path, generate_batch_completion,
        format_tabs=True
    )
