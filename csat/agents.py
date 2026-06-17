"""Agents producing an action string from the message history.

HumanAgent -- you type the action (for --human roleplay; needs no torch).
ModelAgent -- loads an HF causal LM, generates on-policy, optionally captures
              residual-stream activations at the decision token.

Generation is fully on-policy: the model emits every token. An action
stop-criterion halts the instant one COMPLETE action line (SET.../SUBMIT + a
newline) appears, so the model can't run past its own decision into a
hallucinated re-evaluation. Truncating its own output is on-policy; if it never
reaches an action within cfg.max_new_tokens that is recorded as an honest
outcome rather than patched over.

NOTE: defaults to AutoModelForCausalLM (e.g. gemma-2-9b-it, a text model). For a
multimodal checkpoint (e.g. gemma-3), swap to AutoModelForImageTextToText +
AutoProcessor as in your original agents.py -- the rest is unchanged.
"""
from __future__ import annotations
import os


class HumanAgent:
    is_model = False

    def act(self, messages, capture_path=None):
        print("\n" + "=" * 70)
        print(messages[-1]["content"])
        print("=" * 70)
        return input("your action > ").strip(), {"source": "human"}


class ModelAgent:
    is_model = True

    def __init__(self, cfg):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, dtype=torch.bfloat16, device_map="auto")
        self.model.eval()

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
        ids = self.tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        return ids.to(self.model.device)

    def _action_stopper(self, prompt_len):
        """Halt as soon as one COMPLETE action line (newline-terminated
        SET.../SUBMIT) appears. Only truncates the model's own tokens -> on-policy."""
        from transformers import StoppingCriteria, StoppingCriteriaList
        import torch
        from .dsl import parse_action
        tok = self.tok

        class _ActionStop(StoppingCriteria):
            def __call__(self, input_ids, scores=None, **kw):
                done = []
                for row in input_ids:
                    hit = False
                    tail = tok.decode(row[-1:].tolist(), skip_special_tokens=True)
                    if "\n" in tail:                              # cheap gate
                        text = tok.decode(row[prompt_len:], skip_special_tokens=True)
                        for ln in text.split("\n")[:-1]:          # complete lines only
                            if parse_action(ln).kind in ("set", "submit"):
                                hit = True; break
                    done.append(hit)
                return torch.tensor(done, dtype=torch.bool, device=input_ids.device)

        return StoppingCriteriaList([_ActionStop()])

    def _generate(self, ids, stopper):
        import torch
        return self.model.generate(
            ids, max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.temperature > 0, temperature=self.cfg.temperature,
            pad_token_id=self.tok.eos_token_id,
            attention_mask=torch.ones_like(ids),
            stopping_criteria=stopper)

    def act(self, messages, capture_path=None):
        ids = self._build_inputs(messages)
        prompt_len = ids.shape[1]
        full = self._generate(ids, self._action_stopper(prompt_len))
        text = self.tok.decode(full[0, prompt_len:], skip_special_tokens=True).strip()
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
