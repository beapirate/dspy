import logging
import random
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
from dspy.utils.callback import with_callbacks


logger = logging.getLogger(__name__)


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
            name="reasoning",
            field=rationale_field,
            type_=rationale_field_type,
        )
        self.reasoning_key = reasoning_key

        self.config = config
        self.stage = random.randbytes(8).hex()

        self.lm = None
        self.demos = []

    def _forward_preprocess(self, **kwargs):
        assert "new_signature" not in kwargs, "new_signature is no longer a valid keyword argument."
        signature = ensure_signature(kwargs.pop("signature", self.signature))
        demos = kwargs.pop("demos", self.demos)
        config = dict(**self.config, **kwargs.pop("config", {}))

        lm = kwargs.pop("lm", self.lm) or settings.lm
        assert isinstance(lm, BaseLM), "No LM is loaded."

        temperature = config.get("temperature") or lm.kwargs.get("temperature")
        num_generations = config.get("n") or lm.kwargs.get("n") or lm.kwargs.get("num_generations") or 1
        if (temperature is None or temperature <= 0.15) and num_generations > 1:
            config["temperature"] = 0.7

        if not all(k in kwargs for k in signature.input_fields):
            present = [k for k in signature.input_fields if k in kwargs]
            missing = [k for k in signature.input_fields if k not in kwargs]
            logger.warning(
                "Not all input fields were provided to module. Present: %s. Missing: %s.",
                present,
                missing,
            )

        return lm, config, signature, demos, kwargs

    def _forward_postprocess(self, completions, signature, **kwargs):
        pred = Prediction.from_completions(completions, signature=signature)
        if kwargs.pop("_trace", True) and settings.trace is not None:
            settings.trace.append((self, {**kwargs}, pred))
        return pred

    def _should_stream(self):
        stream_listeners = settings.stream_listeners or []
        should_stream = settings.send_stream is not None
        if should_stream and len(stream_listeners) > 0:
            should_stream = any(sl.predict == self for sl in stream_listeners)
        return should_stream

    @with_callbacks
    def _lm_forward(self, lm, **kwargs):
        return lm.forward(**kwargs)

    def forward(self, **kwargs):
        lm, config, signature, demos, kwargs = self._forward_preprocess(**kwargs)

        adapter = settings.adapter or ChatAdapter()

        messages = adapter.format(signature, demos, kwargs)

        if self._should_stream():
            with settings.context(caller_predict=self):
                response = self._lm_forward(lm, messages=messages, **config)
        else:
            with settings.context(send_stream=None):
                response = self._lm_forward(lm, messages=messages, **config)

        outputs = lm._process_lm_response(response, None, messages, **config)
        completions = adapter._call_post_process(outputs, signature)

        reasoning = []
        for choice in getattr(response, "choices", []):
            r = None
            if hasattr(choice, "message") and hasattr(choice.message, self.reasoning_key):
                r = getattr(choice.message, self.reasoning_key)
            elif hasattr(choice, self.reasoning_key):
                r = getattr(choice, self.reasoning_key)
            elif isinstance(choice, dict):
                r = choice.get(self.reasoning_key)
                if r is None and "message" in choice:
                    r = choice["message"].get(self.reasoning_key)
            reasoning.append(r)

        for comp, r in zip(completions, reasoning or []):
            comp["reasoning"] = r

        prediction = self._forward_postprocess(completions, self.extended_signature, **kwargs)
        return prediction
