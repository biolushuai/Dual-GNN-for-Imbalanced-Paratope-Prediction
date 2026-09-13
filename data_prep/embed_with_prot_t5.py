from __future__ import annotations

import argparse
import gc
import os
import pickle
import re
from typing import List, Sequence

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer


def sanitise(sequence: str) -> str:
    """Mirror ProtT5 preprocessing: rare residues -> X, residues space-separated."""
    sequence = re.sub(r"[UZOB]", "X", sequence)
    return " ".join(list(sequence))


@torch.no_grad()
def embed_sequences(
    sequences: Sequence[str],
    model: T5EncoderModel,
    tokenizer: T5Tokenizer,
    device: torch.device,
    batch_size: int = 8,
) -> List[np.ndarray]:
    """Run the encoder on a list of sequences and return one numpy array per sequence."""
    out: List[np.ndarray] = []
    batches = [sequences[i : i + batch_size] for i in range(0, len(sequences), batch_size)]
    for batch_idx, batch in enumerate(batches):
        print(f"[ProtT5] batch {batch_idx + 1}/{len(batches)} (size={len(batch)})")
        encoded = tokenizer.batch_encode_plus(batch, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(encoded["input_ids"], device=device)
        attention_mask = torch.tensor(encoded["attention_mask"], device=device)
        hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = hidden.cpu().numpy()
        for seq_idx in range(len(hidden)):
            seq_len = int(attention_mask[seq_idx].sum().item())
            # Drop the trailing </s> token so the embedding aligns with residues.
            out.append(hidden[seq_idx][: seq_len - 1].astype(np.float16))
        del input_ids, attention_mask, hidden
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Append ProtT5 embeddings to a PECAN pickle.")
    parser.add_argument("--input", required=True,
                        help="Pickle produced by build_structure_dataset.py")
    parser.add_argument("--output", required=True,
                        help="Where to write the augmented pickle")
    parser.add_argument("--model-name-or-path", default="Rostlab/prot_t5_xl_half_uniref50-enc")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sequence-key", default="antibody_sequence")
    parser.add_argument("--feature-key", default="ab_feature")
    args = parser.parse_args(argv)

    with open(args.input, "rb") as handle:
        data = pickle.load(handle)

    sequences = [sanitise(record[args.sequence_key]) for record in data]
    print(f"Loaded {len(data)} records; embedding with {args.model_name_or_path}")

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16).to(args.device)
    model.eval()

    embeddings = embed_sequences(sequences, model, tokenizer, torch.device(args.device),
                                 batch_size=args.batch_size)
    if len(embeddings) != len(data):
        raise RuntimeError(
            f"Embedding count ({len(embeddings)}) does not match record count ({len(data)})."
        )
    for record, embedding in zip(data, embeddings):
        record[args.feature_key] = embedding

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump(data, handle)
    print(f"Wrote {args.output} with embeddings shape {embeddings[0].shape}")


if __name__ == "__main__":
    main()
