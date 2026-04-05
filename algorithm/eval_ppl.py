"""Evaluate wikitext-2 perplexity for HuggingFace AQLM models."""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import trange
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_loader import load_aqlm_model


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, dataset_name="wikitext2", seqlen=4096):
    if dataset_name == "wikitext2":
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen
    device = next(model.parameters()).device

    nlls = []
    for i in trange(nsamples, desc=f"Evaluating {dataset_name} perplexity"):
        batch = testenc[:, i * seqlen : (i + 1) * seqlen].to(device)
        outputs = model(batch)
        lm_logits = outputs.logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss.float() * seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()
    return ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    model = load_aqlm_model(args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    ppl = evaluate_perplexity(model, tokenizer, "wikitext2", args.seqlen)
    print(f"\nwikitext2 perplexity = {ppl:.4f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"model": args.model_path, "dataset": "wikitext2", "perplexity": ppl}, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
