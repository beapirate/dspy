from typing import Optional, Type, Union

from pydantic.fields import FieldInfo

import dspy
from dspy.clients.base_lm import BaseLM
from dspy.dsp.utils import settings
from dspy.primitives.prediction import Prediction
from dspy.primitives.program import Module
from dspy.signatures.field import OutputField
from dspy.signatures.signature import Signature, ensure_signature
from dspy.utils.callback import BaseCallback


class NativeChainOfThought(Module):
    """Run a task using an LM that natively returns its reasoning."""

    def __init__(
        self,
        signature: Type[Signature],
        rationale_field: Optional[Union[OutputField, FieldInfo]] = None,
        rationale_field_type: Type = str,
        reasoning_key: str = "reasoning_output",
        **config,
    ):
        super().__init__()
        signature = ensure_signature(signature)

        rationale_field_type = rationale_field.annotation if rationale_field else rationale_field_type
        rationale_field = rationale_field if rationale_field else dspy.OutputField()

        self.signature = signature
        self.extended_signature = signature.prepend(
            name="reasoning", field=rationale_field, type_=rationale_field_type
        )
        self.reasoning_key = reasoning_key
        self.predict = dspy.Predict(signature, **config)
        self.demos = []
        self.lm = None

    class _CaptureLMCallback(BaseCallback):
        def __init__(self):
            self.response = None

        def on_lm_end(self, call_id, outputs, exception=None):
            self.response = outputs

    def forward(self, **kwargs):
        signature = ensure_signature(kwargs.pop("signature", self.signature))
        demos = kwargs.pop("demos", self.demos)
        config = kwargs.pop("config", {})
        lm = kwargs.get("lm", self.lm)
        lm_to_use = lm or self.predict.lm or settings.lm
        assert isinstance(lm_to_use, BaseLM), "No LM is loaded."

        capture_cb = self._CaptureLMCallback()
        overrides = {"callbacks": settings.get("callbacks", []) + [capture_cb], "lm": lm_to_use}

        with settings.context(**overrides):
            pred = self.predict(
                _trace=False, signature=signature, demos=demos, config=config, **kwargs
            )

        completions = {k: list(v) for k, v in pred.completions.items()}
        completions["reasoning"] = []
        response = capture_cb.response
        if response is not None:
            for choice in response.choices:
                reasoning = None
                if hasattr(choice, "message") and hasattr(choice.message, self.reasoning_key):
                    reasoning = getattr(choice.message, self.reasoning_key)
                elif hasattr(choice, self.reasoning_key):
                    reasoning = getattr(choice, self.reasoning_key)
                elif isinstance(choice, dict):
                    reasoning = choice.get(self.reasoning_key)
                    if reasoning is None and "message" in choice:
                        reasoning = choice["message"].get(self.reasoning_key)
                completions["reasoning"].append(reasoning)
        else:
            completions["reasoning"] = [None for _ in range(len(next(iter(completions.values()), [])))]

        prediction = Prediction.from_completions(completions, signature=self.extended_signature)
        prediction.set_lm_usage(pred.get_lm_usage())

        if kwargs.pop("_trace", True) and settings.trace is not None:
            settings.trace.append((self, {**kwargs}, prediction))

        return prediction
