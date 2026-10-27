"""Model-derived visual observations remain distinguishable from verbatim source text."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class VisualValue(BaseModel):
    model_config = ConfigDict(extra='forbid')
    label: str
    value: str
    unit: str | None
    evidence_text: str


class VisualObservation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kind: Literal['chart', 'table', 'diagram', 'photo', 'text', 'logo', 'other']
    title: str | None
    x_axis: str | None
    y_axis: str | None
    legend: list[str]
    visible_text: list[str]
    numeric_values: list[VisualValue]
    observations: list[str]
    interpretation: str = Field(min_length=1)
    limitations: list[str]
    status: Literal['interpreted', 'insufficient_detail']

    @model_validator(mode='after')
    def printed_numbers_only(self):
        # Consistency validation, not proof that model-reported text matches pixels.
        visible = ' '.join(self.visible_text).casefold()
        for item in self.numeric_values:
            if not item.value.strip() or item.value.casefold() not in item.evidence_text.casefold():
                raise ValueError('A numeric value must quote its printed evidence.')
            if item.evidence_text.casefold() not in visible:
                raise ValueError('Numeric evidence must occur in transcribed visible text.')
        return self


class VisualDescription(BaseModel):
    model: str
    prompt_version: str
    image_sha256: str
    model_derived: Literal[True] = True
    observation: VisualObservation
