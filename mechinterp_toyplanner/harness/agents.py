"""Agents that produce an action string given the message history.

HumanAgent  -- you type the action (for --human roleplay).
ModelAgent  -- loads the HF model, generates, optionally captures activations.

torch/transformers are imported lazily inside ModelAgent so --human needs neither.

Generation is fully on-policy: the model produces every token itself. The only
control is an action stop-criteria that halts the instant one COMPLETE action
line (SET.../SUBMIT terminated by a newline) is emitted, so the model can't run
PAST its decision into a hallucinated 're-evaluation'. Truncating the model's own
output is on-policy (no tokens are inserted); if it never reaches an action
within cfg.max_new_tokens, that is recorded as an honest outcome rather than
patched over.
"""
from __future__ import annotations
import os


class HumanAgent:
    is_model = False

    def act(self, messages, capture_path=None):
        # show the latest environment turn, then read one action line
        print("\n" + "=" * 70)
        print(messages[-1]["content"])
        print("=" * 70)
        text = input("your action > ").strip()
        return text, {"source": "human"}


class ModelAgent:
    is_model = True

    def __init__(self, cfg):
        import torch
        self.cfg = cfg
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, dtype=torch.bfloat16, device_map=cfg.device)
            
        """

        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.model = AutoModelForImageTextToText.from_pretrained(
            cfg.model_name,
            dtype=torch.bfloat16,
            device_map="auto",          # or cfg.device if you know it is supported
        )

        self.processor = AutoProcessor.from_pretrained(cfg.model_name)

        self.model.eval()
        self.device = cfg.device

    def _build_inputs(self, messages):
        # Gemma has no system role -> fold system text into the first user turn.
        msgs, sys_txt = [], None
        for m in messages:
            if m["role"] == "system":
                sys_txt = m["content"]; continue
            if sys_txt and m["role"] == "user":
                m = {"role": "user", "content": sys_txt + "\n\n" + m["content"]}
                sys_txt = None
            msgs.append(m)
        out = self.processor.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        

        if hasattr(out, "keys"):
            out = out["input_ids"]

        return out.to(self.model.device)

  
    def _action_stopper(self, prompt_len):
        """StoppingCriteria that halts as soon as one COMPLETE action line
        (newline-terminated SET.../SUBMIT) appears in the generated text.
        This only truncates the model's own tokens -- it is on-policy."""
        from transformers import StoppingCriteria, StoppingCriteriaList
        import torch
        from .dsl import parse_action
        tok = self.processor

        class _ActionStop(StoppingCriteria):
            def __call__(self, input_ids, scores=None, **kw):
                done = []
                for row in input_ids:
                    hit = False
                    # cheap gate: only inspect lines when the latest token ended one
                    tail = tok.decode(row[-1:].tolist(), skip_special_tokens=True)
                    if "\n" in tail:
                        text = tok.decode(row[prompt_len:], skip_special_tokens=True)
                        for ln in text.split("\n")[:-1]:        # complete lines only
                            if parse_action(ln).kind in ("set", "submit"):
                                hit = True
                                break
                    done.append(hit)
                return torch.tensor(done, dtype=torch.bool, device=input_ids.device)

        return StoppingCriteriaList([_ActionStop()])

    def _generate(self, ids, max_new, stopper):
        import torch
        return self.model.generate(
            ids, max_new_tokens=max_new,
            do_sample=self.cfg.temperature > 0, temperature=self.cfg.temperature,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            attention_mask=torch.ones_like(ids),     # silences the attn-mask warning
            stopping_criteria=stopper)

    def act(self, messages, capture_path=None):
        ids = self._build_inputs(messages)
        prompt_len = ids.shape[1]

        # single on-policy pass; stop at the first complete action line
        full = self._generate(ids, self.cfg.max_new_tokens,
                              self._action_stopper(prompt_len))

        text = self.processor.decode(full[0, prompt_len:], skip_special_tokens=True).strip()
        meta = {"source": "model", "prompt_len": int(prompt_len),
                "resp_len": int(full.shape[1] - prompt_len),
                "temperature": self.cfg.temperature}
        if capture_path and self.cfg.capture:
            from .recorder import capture_and_save
            capture_and_save(self.model, full, prompt_len, capture_path,
                             tokens=self.cfg.capture_tokens,
                             last_k=getattr(self.cfg, "capture_last_k", 5),
                             dtype=self.cfg.capture_dtype)
            meta["activations"] = os.path.basename(capture_path)
        return text, meta