"""Policy schema models for breeding-program runs."""

from __future__ import annotations

from typing import Dict, Literal, Optional

try:
    from pydantic import BaseModel, Field
except Exception:
    class BaseModel:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self):
            out = {}
            for k, v in self.__dict__.items():
                if hasattr(v, "model_dump"):
                    out[k] = v.model_dump()
                else:
                    out[k] = v
            return out

    def Field(default=None, **kwargs):  # type: ignore
        if "default_factory" in kwargs:
            return kwargs["default_factory"]()
        return default


class Step3Policy(BaseModel):
    caller: Literal["gatk_gvcf_joint"] = "gatk_gvcf_joint"


class Step4Policy(BaseModel):
    min_sample_callrate: float = 0.95
    min_snp_callrate: float = 0.98
    min_maf: float = 0.01
    enable_hwe: bool = False
    hwe_p: float = 1e-6
    het_outlier_method: Literal["MAD", "Z"] = "MAD"
    het_outlier_thresh: float = 3.5


class Step5Policy(BaseModel):
    enabled: bool = True
    method: Literal["beagle"] = "beagle"


class ModelingPolicy(BaseModel):
    default_model: Literal["gblup"] = "gblup"
    enable_ssgblup_if_pedigree: bool = True
    cv_mode_default: Literal["forward_time", "random_kfold"] = "forward_time"
    rnd_enabled_by_default: bool = False


class PolicyModel(BaseModel):
    species: str
    goal: str
    reference_build: str = Field(..., description="Reference genome build")
    step3: Step3Policy = Step3Policy()
    step4: Step4Policy = Step4Policy()
    step5: Step5Policy = Step5Policy()
    modeling: ModelingPolicy = ModelingPolicy()
    rnd: Dict[str, object] = Field(default_factory=dict)
    extras: Dict[str, object] = Field(default_factory=dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(getattr(self, "step3", None), Step3Policy):
            self.step3 = Step3Policy(**(getattr(self, "step3", {}) or {}))
        if not isinstance(getattr(self, "step4", None), Step4Policy):
            self.step4 = Step4Policy(**(getattr(self, "step4", {}) or {}))
        if not isinstance(getattr(self, "step5", None), Step5Policy):
            self.step5 = Step5Policy(**(getattr(self, "step5", {}) or {}))
        if not isinstance(getattr(self, "modeling", None), ModelingPolicy):
            self.modeling = ModelingPolicy(**(getattr(self, "modeling", {}) or {}))

