"""
SFT Model Inference
===================

Generate medical summaries using the Stage A fine-tuned model.

Usage (PowerShell friendly):
    python sft_inference.py --model_path .\models\sft_specialist\final_model --clinical_note "Patient reports fever..."
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List

import torch

# Optional imports; if you don't use PEFT or tokenizers package, install them first:
# pip install peft tokenizers
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# tokenizers is the Hugging Face Rust-backed tokenizers library
try:
    import tokenizers as _tokenizers_lib  # used for tokenizer.json fallback
except Exception:
    _tokenizers_lib = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SFTInference:
    """
    Inference wrapper for fine-tuned SFT models.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: str = "float16",
        use_8bit: bool = False,
        trust_remote_code: bool = False,
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to the fine-tuned model (local directory)
            device: Device to use (e.g. "cuda", "cpu", "cuda:0")
            torch_dtype: Data type for model (float16, bfloat16, float32)
            use_8bit: Whether to use 8-bit quantization
            trust_remote_code: Pass to from_pretrained if model uses custom code
        """
        # Normalize and resolve path (use POSIX style for HF internals on Windows)
        model_path = Path(model_path).resolve()
        self.model_path = str(model_path)
        self.model_path_posix = model_path.as_posix()

        self.device_str = device if device is not None else "cpu"
        self.device = torch.device(self.device_str)

        # Convert dtype string to torch dtype
        dtype_name = torch_dtype.replace("torch.", "")
        if not hasattr(torch, dtype_name):
            raise ValueError(f"Unknown torch dtype: {torch_dtype}")
        self.torch_dtype = getattr(torch, dtype_name)

        logger.info(f"Loading model from {self.model_path}")
        logger.info(f"Using device: {self.device} (device_str='{self.device_str}'), dtype={self.torch_dtype}, use_8bit={use_8bit}")

        # Load tokenizer (local-only). If AutoTokenizer fails due to hub validation,
        # try robust local fallbacks: tokenizer.json (fast) or sentencepiece.
        self.tokenizer = None
        try:
            # Use posix-style path to reduce HF hub repo-id parsing issues on Windows
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path_posix,
                local_files_only=True,
                trust_remote_code=trust_remote_code,
            )
            logger.info("Loaded tokenizer with AutoTokenizer.from_pretrained(local_files_only=True).")
        except Exception as e_auto:
            logger.warning(f"AutoTokenizer.from_pretrained(local) failed: {e_auto!r}")
            logger.info("Attempting manual tokenizer fallback (tokenizer.json or tokenizer.model)...")

            # Try tokenizer.json (fast) with tokenizers lib
            tok_json = model_path / "tokenizer.json"
            tok_sp_model = model_path / "tokenizer.model"  # sentencepiece / spiece
            tok_config = model_path / "tokenizer_config.json"

            if tok_json.exists() and _tokenizers_lib is not None:
                try:
                    tokenizer_obj = _tokenizers_lib.Tokenizer.from_file(str(tok_json))
                    # Wrap in PreTrainedTokenizerFast
                    self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
                    logger.info("Loaded tokenizer from tokenizer.json via tokenizers + PreTrainedTokenizerFast.")
                    # if tokenizer_config.json exists, try to set special tokens
                    if tok_config.exists():
                        try:
                            cfg = json.load(open(str(tok_config), "r", encoding="utf-8"))
                            for tokname in ("bos_token", "eos_token", "unk_token", "pad_token", "sep_token", "cls_token"):
                                if tokname in cfg and cfg[tokname]:
                                    setattr(self.tokenizer, tokname, cfg[tokname])
                        except Exception:
                            pass
                except Exception as e_fallback:
                    logger.error(f"Failed to load tokenizer.json fallback: {e_fallback!r}")
                    raise RuntimeError(f"Failed tokenizer.json fallback: {e_fallback}") from e_fallback

            elif tok_sp_model.exists():
                # SentencePiece fallback: let AutoTokenizer attempt to load again pointing at dir (may still fail)
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_path_posix,
                        local_files_only=True,
                        trust_remote_code=trust_remote_code,
                    )
                    logger.info("Loaded tokenizer via AutoTokenizer (sentencepiece fallback).")
                except Exception as e_sp:
                    logger.error(f"SentencePiece AutoTokenizer fallback failed: {e_sp!r}")
                    raise RuntimeError(f"Failed to load sentencepiece tokenizer from {model_path}") from e_sp
            else:
                logger.error(f"No tokenizer.json or tokenizer.model found in {model_path}")
                logger.error("Make sure you ran tokenizer.save_pretrained(...) when saving the model directory.")
                raise RuntimeError(f"Tokenizer files missing in {model_path}") from e_auto

        # Ensure there is a pad token
        if getattr(self.tokenizer, "pad_token", None) is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # Add a pad token if neither pad nor eos exist
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        if getattr(self.tokenizer, "pad_token_id", None) is None:
            # set pad_token_id from pad_token or eos_token
            if self.tokenizer.pad_token is not None:
                self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
            elif self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)

        # Model load kwargs and device_map
        load_kwargs = dict(
            torch_dtype=self.torch_dtype,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        if use_8bit:
            load_kwargs["load_in_8bit"] = True

        # device_map for local loading: some HF versions accept a dict mapping "" -> "cuda:0" etc.
        device_map = {"": self.device_str}

        # Try to load as PEFT (LoRA) model first, then fallback to AutoModelForCausalLM
        self.model = None
        try:
            logger.info("Attempting to load model as PEFT/LoRA via AutoPeftModelForCausalLM...")
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                self.model_path_posix,
                device_map=device_map,
                **load_kwargs,
            )
            logger.info("Loaded model using AutoPeftModelForCausalLM (PEFT/LoRA).")
        except Exception as e_peft:
            logger.warning(f"PEFT load failed: {e_peft!r}")
            logger.info("Falling back to AutoModelForCausalLM.from_pretrained() ...")
            try:
                from transformers import AutoModelForCausalLM

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path_posix,
                    device_map=device_map,
                    **load_kwargs,
                )
                logger.info("Loaded model using AutoModelForCausalLM.")
            except Exception as e_base:
                logger.error(f"Failed to load model from local path '{self.model_path}': {e_base!r}")
                logger.error("If you intended to load from Hugging Face hub, pass a repo id (e.g. 'namespace/repo') and set local_files_only=False.")
                raise RuntimeError(f"Model loading failed: {e_base}") from e_base

        # If model not already on desired device, try `.to(...)`
        try:
            self.model.to(self.device)
        except Exception:
            # ignore if model is device-mapped already
            pass

        self.model.eval()
        logger.info("Model loaded and set to eval mode")

    def generate_summary(
        self,
        clinical_note: str,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> Dict[str, str]:
        """
        Generate a summary for a clinical note.

        Returns a dict with keys:
            - clinical_note
            - prompt
            - generated_summary
            - full_output
        """
        prompt = f"Clinical Note: {clinical_note}\n\nSummary:"

        encodings = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        # Move tensors to device
        encodings = {k: v.to(self.device) for k, v in encodings.items() if isinstance(v, torch.Tensor)}

        input_ids = encodings["input_ids"]
        attention_mask = encodings.get("attention_mask", None)

        logger.info(f"Input tokens: {input_ids.shape[1]}")

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
            )

        # Decode first example in batch
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from generated_text if present
        summary = generated_text
        if generated_text.startswith(prompt):
            summary = generated_text[len(prompt) :].strip()
        else:
            summary = generated_text.replace(prompt, "").strip()

        return {
            "clinical_note": clinical_note,
            "prompt": prompt,
            "generated_summary": summary,
            "full_output": generated_text,
        }

    def batch_generate(
        self,
        clinical_notes: List[str],
        max_new_tokens: int = 150,
        temperature: float = 0.7,
    ) -> List[Dict[str, str]]:
        results = []
        for note in clinical_notes:
            results.append(
                self.generate_summary(
                    note,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            )
        return results


def main():
    parser = argparse.ArgumentParser(description="SFT Model Inference")

    parser.add_argument("--model_path", required=True, help="Path to the fine-tuned model (local directory)")
    parser.add_argument("--clinical_note", type=str, help="Clinical note to summarize")
    parser.add_argument("--input_file", type=str, help="Path to file with clinical notes (one per line)")
    parser.add_argument("--max_tokens", type=int, default=150, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda or cpu or cuda:0)")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--trust_remote_code", action="store_true", help="Allow loading model/tokenizer code from the repo (only for hub loads)")

    args = parser.parse_args()

    # Validate model path exists (we default to local_files_only)
    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        logger.error("If you want to load from the Hugging Face Hub, pass a repo id and set local_files_only=False in the code.")
        raise SystemExit(1)

    inference = SFTInference(
        model_path=str(model_path),
        device=args.device,
        use_8bit=args.use_8bit,
        trust_remote_code=args.trust_remote_code,
    )

    # Single note
    if args.clinical_note:
        logger.info("Generating summary for single clinical note...")
        result = inference.generate_summary(
            args.clinical_note,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\n{'='*60}")
        print(f"Clinical Note:\n{result['clinical_note']}")
        print(f"\n{'-'*60}")
        print(f"Generated Summary:\n{result['generated_summary']}")
        print(f"{'='*60}\n")

    # File input
    elif args.input_file:
        logger.info(f"Reading clinical notes from {args.input_file}")
        with open(args.input_file, "r", encoding="utf-8") as f:
            notes = [line.strip() for line in f if line.strip()]
        logger.info(f"Generating summaries for {len(notes)} notes...")
        results = inference.batch_generate(notes, max_new_tokens=args.max_tokens, temperature=args.temperature)
        for i, result in enumerate(results, 1):
            print(f"\n{'='*60}")
            print(f"Example {i}")
            print(f"Clinical Note:\n{result['clinical_note']}")
            print(f"\n{'-'*60}")
            print(f"Generated Summary:\n{result['generated_summary']}")
            print(f"{'='*60}\n")

    # Interactive mode
    else:
        print("SFT Inference - Interactive Mode")
        print("Type clinical notes and press Enter twice to generate summaries")
        print("Type 'quit' to exit\n")

        while True:
            print("Enter clinical note (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line.strip().lower() == "quit":
                    return
                if line == "":
                    if lines:
                        break
                else:
                    lines.append(line)
            clinical_note = " ".join(lines)
            result = inference.generate_summary(
                clinical_note,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(f"\nGenerated Summary:\n{result['generated_summary']}\n")


if __name__ == "__main__":
    main()
