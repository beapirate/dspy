from typing import Optional, Type, Union

from pydantic.fields import FieldInfo

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.base_lm import BaseLM
from dspy.dsp.utils import settings
from dspy.primitives.prediction import Prediction
from dspy.primitives.program import Module
from dspy.signatures.field import OutputField
from dspy.signatures.signature import Signature, ensure_signature


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
        self.extended_signature = signature.prepend(name="reasoning", field=rationale_field, type_=rationale_field_type)
        self.reasoning_key = reasoning_key
        self.config = config
        self.demos = []
        self.lm = None

    def forward(self, **kwargs):
        signature = ensure_signature(kwargs.pop("signature", self.signature))
        demos = kwargs.pop("demos", self.demos)
        config = dict(**self.config, **kwargs.pop("config", {}))
        lm = kwargs.pop("lm", self.lm) or settings.lm

        assert isinstance(lm, BaseLM), "No LM is loaded."

        adapter = settings.adapter or ChatAdapter()

        messages = adapter.format(signature, demos, kwargs)
        response = lm.forward(messages=messages, **config)

        completions = {k: [] for k in signature.output_fields}
        completions["reasoning"] = []

        for choice in response.choices:
            completion_text = choice.message.content if hasattr(choice, "message") else choice.get("text", "")
            parsed = adapter.parse(signature, completion_text)
            for field, value in parsed.items():
                completions[field].append(value)

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

        prediction = Prediction.from_completions(completions, signature=self.extended_signature)

        if kwargs.pop("_trace", True) and settings.trace is not None:
            settings.trace.append((self, {**kwargs}, prediction))

        return prediction
