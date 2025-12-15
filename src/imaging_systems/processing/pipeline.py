"""Image processing pipelines."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
import numpy as np
import copy

from .filters import ImageFilter


class StepStatus(Enum):
    """Processing step status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStep:
    """A single step in the processing pipeline."""

    name: str
    processor: Callable[[np.ndarray], np.ndarray]
    enabled: bool = True
    parameters: dict = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    execution_time: float = 0.0
    error_message: str = ""

    def execute(self, image: np.ndarray) -> np.ndarray:
        """Execute the processing step."""
        import time

        self.status = StepStatus.RUNNING
        start_time = time.time()

        try:
            if isinstance(self.processor, ImageFilter):
                result = self.processor.apply(image)
            else:
                result = self.processor(image, **self.parameters)

            self.execution_time = time.time() - start_time
            self.status = StepStatus.COMPLETED
            return result

        except Exception as e:
            self.status = StepStatus.FAILED
            self.error_message = str(e)
            self.execution_time = time.time() - start_time
            raise


@dataclass
class PipelineResult:
    """Result of pipeline execution."""

    success: bool
    input_image: np.ndarray
    output_image: np.ndarray
    intermediate_results: list[np.ndarray]
    step_results: list[dict]
    total_execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


class ProcessingPipeline:
    """Configurable image processing pipeline."""

    def __init__(self, name: str = "default"):
        self.name = name
        self._steps: list[PipelineStep] = []
        self._save_intermediate = False
        self._stop_on_error = True

    def add_step(
        self,
        name: str,
        processor: Callable[[np.ndarray], np.ndarray],
        parameters: dict = None,
        position: int = -1,
    ) -> "ProcessingPipeline":
        """Add a processing step to the pipeline."""
        step = PipelineStep(
            name=name,
            processor=processor,
            parameters=parameters or {},
        )

        if position < 0:
            self._steps.append(step)
        else:
            self._steps.insert(position, step)

        return self

    def add_filter(
        self,
        name: str,
        filter_instance: ImageFilter,
        position: int = -1,
    ) -> "ProcessingPipeline":
        """Add a filter step to the pipeline."""
        return self.add_step(name, filter_instance, position=position)

    def remove_step(self, name: str) -> bool:
        """Remove a step by name."""
        for i, step in enumerate(self._steps):
            if step.name == name:
                self._steps.pop(i)
                return True
        return False

    def enable_step(self, name: str, enabled: bool = True):
        """Enable or disable a step."""
        for step in self._steps:
            if step.name == name:
                step.enabled = enabled
                break

    def reorder_steps(self, order: list[str]):
        """Reorder steps by name list."""
        step_dict = {s.name: s for s in self._steps}
        self._steps = [step_dict[name] for name in order if name in step_dict]

    def set_save_intermediate(self, save: bool):
        """Set whether to save intermediate results."""
        self._save_intermediate = save

    def set_stop_on_error(self, stop: bool):
        """Set whether to stop on error."""
        self._stop_on_error = stop

    def execute(self, image: np.ndarray) -> PipelineResult:
        """Execute the pipeline on an image."""
        import time

        start_time = time.time()
        current_image = image.copy()
        intermediate_results = []
        step_results = []
        success = True

        for step in self._steps:
            if not step.enabled:
                step.status = StepStatus.SKIPPED
                step_results.append({
                    "name": step.name,
                    "status": "skipped",
                    "execution_time": 0,
                })
                continue

            try:
                current_image = step.execute(current_image)

                if self._save_intermediate:
                    intermediate_results.append(current_image.copy())

                step_results.append({
                    "name": step.name,
                    "status": "completed",
                    "execution_time": step.execution_time,
                })

            except Exception as e:
                step_results.append({
                    "name": step.name,
                    "status": "failed",
                    "error": str(e),
                    "execution_time": step.execution_time,
                })

                if self._stop_on_error:
                    success = False
                    break

        total_time = time.time() - start_time

        return PipelineResult(
            success=success,
            input_image=image,
            output_image=current_image,
            intermediate_results=intermediate_results,
            step_results=step_results,
            total_execution_time=total_time,
            metadata={
                "pipeline_name": self.name,
                "num_steps": len(self._steps),
                "num_executed": len([s for s in step_results if s["status"] != "skipped"]),
            },
        )

    def get_steps(self) -> list[dict]:
        """Get list of steps with their configuration."""
        return [
            {
                "name": step.name,
                "enabled": step.enabled,
                "parameters": step.parameters,
                "status": step.status.value,
            }
            for step in self._steps
        ]

    def copy(self) -> "ProcessingPipeline":
        """Create a copy of the pipeline."""
        new_pipeline = ProcessingPipeline(f"{self.name}_copy")
        new_pipeline._steps = copy.deepcopy(self._steps)
        new_pipeline._save_intermediate = self._save_intermediate
        new_pipeline._stop_on_error = self._stop_on_error
        return new_pipeline

    def reset(self):
        """Reset all step statuses."""
        for step in self._steps:
            step.status = StepStatus.PENDING
            step.execution_time = 0.0
            step.error_message = ""


class PipelineFactory:
    """Factory for creating preset processing pipelines."""

    @staticmethod
    def create_ultrasound_pipeline() -> ProcessingPipeline:
        """Create pipeline optimized for ultrasound images."""
        from .filters import NoiseReductionFilter, ContrastEnhancementFilter, FilterParameters

        pipeline = ProcessingPipeline("ultrasound")

        # Speckle reduction
        pipeline.add_filter(
            "speckle_reduction",
            NoiseReductionFilter("median", FilterParameters(kernel_size=3))
        )

        # Contrast enhancement
        pipeline.add_filter(
            "contrast_enhancement",
            ContrastEnhancementFilter("clahe", FilterParameters(strength=0.5))
        )

        return pipeline

    @staticmethod
    def create_xray_pipeline() -> ProcessingPipeline:
        """Create pipeline optimized for X-ray images."""
        from .filters import NoiseReductionFilter, ContrastEnhancementFilter, EdgeEnhancementFilter, FilterParameters

        pipeline = ProcessingPipeline("xray")

        # Noise reduction
        pipeline.add_filter(
            "noise_reduction",
            NoiseReductionFilter("bilateral", FilterParameters(sigma=1.0))
        )

        # Contrast enhancement
        pipeline.add_filter(
            "contrast_enhancement",
            ContrastEnhancementFilter("clahe", FilterParameters(strength=0.7))
        )

        # Edge enhancement
        pipeline.add_filter(
            "edge_enhancement",
            EdgeEnhancementFilter("unsharp_mask", FilterParameters(strength=0.3))
        )

        return pipeline

    @staticmethod
    def create_oct_pipeline() -> ProcessingPipeline:
        """Create pipeline optimized for OCT images."""
        from .filters import NoiseReductionFilter, ContrastEnhancementFilter, FilterParameters

        pipeline = ProcessingPipeline("oct")

        # Speckle reduction
        pipeline.add_filter(
            "speckle_reduction",
            NoiseReductionFilter("median", FilterParameters(kernel_size=3))
        )

        # Contrast enhancement
        pipeline.add_filter(
            "contrast_enhancement",
            ContrastEnhancementFilter("histogram_equalization")
        )

        return pipeline

    @staticmethod
    def create_thermal_pipeline() -> ProcessingPipeline:
        """Create pipeline optimized for thermal images."""
        from .filters import SmoothingFilter, ContrastEnhancementFilter, FilterParameters

        pipeline = ProcessingPipeline("thermal")

        # Smoothing
        pipeline.add_filter(
            "smoothing",
            SmoothingFilter("gaussian", FilterParameters(sigma=0.5))
        )

        # Contrast enhancement
        pipeline.add_filter(
            "contrast_enhancement",
            ContrastEnhancementFilter("gamma", FilterParameters(strength=0.3))
        )

        return pipeline
