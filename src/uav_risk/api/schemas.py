from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from uav_risk.core.contracts import RawSecondaryOverrides, ScenarioRawInput


class AssessmentRequest(BaseModel):
    scenario: Any
    secondary_overrides: Any = Field(default_factory=RawSecondaryOverrides)
    operator_notes: Optional[str] = None

    @model_validator(mode="after")
    def validate_raw_contracts(self) -> "AssessmentRequest":
        self.scenario = ScenarioRawInput.model_validate(self.scenario)
        self.secondary_overrides = RawSecondaryOverrides.model_validate(self.secondary_overrides)
        return self


class IssueResponse(BaseModel):
    code: str
    field: str | None = None
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
