"""Agents that produce an action string from the message history.

HumanAgent    -- you type the action (roleplay / debugging).
ScriptedAgent -- replays a fixed list of action strings (loop testing, no model).
ModelAgent    -- loads the HF vision model, generates on-policy, optionally
                 captures activations. Each user turn carries an IMAGE (the clean
                 phantom at case start, the dose wash thereafter); images are
                 referenced by `image_path` on the message and loaded here.

torch/transformers are imported lazily, so Human/Scripted paths need neither.
Generation is on-policy: an action stop-criteria halts the instant one complete
action line (SET.../SUBMIT) is emitted, so the model can't run past its decision.
"""
from __future__ import annotations
import os


class HumanAgent:
    is_model = False

    def act(self, messages, capture_path=None):
        print("\n" + "=" * 70)
        print(messages[-1].get("content"))
        if messages[-1].get("image_path"):
            print(f"[image: {messages[-1]['image_path']}]")
        print("=" * 70)
        text = input("your action > ").strip()
        return text, {"source": "human"}


class ScriptedAgent:
    """Replays canned actions; for exercising the env loop without a model."""
    is_model = False

    def __init__(self, actions):
        self.actions = list(actions)
        self.i = 0

    def act(self, messages, capture_path=None):
        a = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return a, {"source": "scripted"}


class ModelAgent:
    is_model = True

    def __init__(self, cfg):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.cfg = cfg
        self.model = AutoModelForImageTextToText.from_pretrained(
            cfg.model_name, dtype=torch.bfloat16, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(cfg.model_name)
        self.model.eval()

    # -- assemble multimodal chat inputs (image per user turn) --------------
    def _build_inputs(self, messages):
        from PIL import Image
        msgs, sys_txt = [], None
        for m in messages:
            if m["role"] == "system":
                sys_txt = m["content"]; continue
            content = []
            text = m["content"]
            if sys_txt and m["role"] == "user":
                text = sys_txt + "\n\n" + text
                sys_txt = None
            if m.get("image_path") and m["role"] == "user":
                content.append({"type": "image",
                                "image": Image.open(m["image_path"]).convert("RGB")})
            content.append({"type": "text", "text": text})
            msgs.append({"role": m["role"], "content": content})
        inputs = self.processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def _action_stopper(self, prompt_len):
        from transformers import StoppingCriteria, StoppingCriteriaList
        import torch
        from .dsl import parse_action
        tok = self.processor

        class _ActionStop(StoppingCriteria):
            def __call__(self, input_ids, scores=None, **kw):
                done = []
                for row in input_ids:
                    hit = False
                    tail = tok.decode(row[-1:].tolist(), skip_special_tokens=True)
                    if "\n" in tail:
                        text = tok.decode(row[prompt_len:], skip_special_tokens=True)
                        for ln in text.split("\n")[:-1]:
                            if parse_action(ln).kind in ("set", "submit"):
                                hit = True; break
                    done.append(hit)
                return torch.tensor(done, dtype=torch.bool, device=input_ids.device)

        return StoppingCriteriaList([_ActionStop()])

    def act(self, messages, capture_path=None):
        import torch
        inputs = self._build_inputs(messages)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            full = self.model.generate(
                **inputs, max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.temperature > 0,
                temperature=self.cfg.temperature,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                stopping_criteria=self._action_stopper(prompt_len))
        text = self.processor.decode(full[0, prompt_len:],
                                     skip_special_tokens=True).strip()
        meta = {"source": "model", "prompt_len": int(prompt_len),
                "resp_len": int(full.shape[1] - prompt_len),
                "temperature": self.cfg.temperature}
        if capture_path and self.cfg.capture:
            from .recorder import capture_and_save
            # re-forward WITHOUT images for a clean text-residual capture is not
            # possible for a VLM; capture the full multimodal forward instead.
            capture_and_save(self.model, full, prompt_len, capture_path,
                             tokens=self.cfg.capture_tokens,
                             last_k=self.cfg.capture_last_k,
                             dtype=self.cfg.capture_dtype)
            meta["activations"] = os.path.basename(capture_path)
        return text, meta
