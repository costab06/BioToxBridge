"""Pure, reusable scientific lifecycle transitions.

No Redis, OAF, artifact, or scientific implementation dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


Transition = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, Any],
]


@dataclass(frozen=True)
class LifecycleStep:
    name: str
    next_stage: str | None
    build_request: Transition | None = None


class Lifecycle:
    def __init__(self, steps: tuple[LifecycleStep, ...]):
        if not steps:
            raise ValueError("Lifecycle requires at least one step")

        names = [step.name for step in steps]

        if len(names) != len(set(names)):
            raise ValueError("Duplicate lifecycle stage")

        self._steps = {step.name: step for step in steps}

        for step in steps:
            if step.next_stage is not None:
                if step.next_stage not in self._steps:
                    raise ValueError(
                        f"Unknown next stage: {step.next_stage}"
                    )

                if step.build_request is None:
                    raise ValueError(
                        f"Missing request builder: {step.name}"
                    )

            elif step.build_request is not None:
                raise ValueError(
                    f"Terminal stage has request builder: {step.name}"
                )

    @property
    def stages(self) -> tuple[str, ...]:
        return tuple(self._steps)

    def next_stage(self, stage: str) -> str | None:
        try:
            return self._steps[stage].next_stage
        except KeyError:
            raise ValueError(
                f"Unexpected lifecycle completion: {stage}"
            ) from None

    def next_request(
        self,
        stage: str,
        output: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> tuple[str | None, dict[str, Any] | None]:

        try:
            step = self._steps[stage]
        except KeyError:
            raise ValueError(
                f"Unexpected lifecycle completion: {stage}"
            ) from None

        if step.next_stage is None:
            return None, None

        assert step.build_request is not None

        request = step.build_request(output, config)

        if not isinstance(request, Mapping):
            raise ValueError("Stage request must be a mapping")

        return step.next_stage, dict(request)
