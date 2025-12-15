"""AI service integration for the imaging system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Callable, Any
import json
import numpy as np

from .inference import ModelConfig, ModelInference, InferenceResult, ModelType, create_model
from .preprocessing import Preprocessor, PreprocessingConfig, InputValidator


class ServiceStatus(Enum):
    """AI service status."""

    OFFLINE = "offline"
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class AIServiceConfig:
    """Configuration for AI service."""

    service_id: str
    name: str
    models: list[ModelConfig] = field(default_factory=list)
    max_batch_size: int = 8
    timeout_seconds: float = 30.0
    auto_load_models: bool = True
    validate_inputs: bool = True
    log_inferences: bool = True
    storage_path: str = "ai_service.json"


@dataclass
class InferenceLog:
    """Log entry for an inference."""

    inference_id: str
    model_id: str
    timestamp: datetime
    input_shape: tuple
    inference_time_ms: float
    result_summary: dict
    success: bool
    error_message: str = ""

    def to_dict(self) -> dict:
        return {
            "inference_id": self.inference_id,
            "model_id": self.model_id,
            "timestamp": self.timestamp.isoformat(),
            "input_shape": self.input_shape,
            "inference_time_ms": self.inference_time_ms,
            "result_summary": self.result_summary,
            "success": self.success,
            "error_message": self.error_message,
        }


class ModelRegistry:
    """Registry for AI models."""

    def __init__(self):
        self._models: dict[str, ModelConfig] = {}
        self._instances: dict[str, ModelInference] = {}

    def register(self, config: ModelConfig):
        """Register a model configuration."""
        self._models[config.model_id] = config

    def unregister(self, model_id: str):
        """Unregister a model."""
        if model_id in self._instances:
            self._instances[model_id].unload()
            del self._instances[model_id]
        if model_id in self._models:
            del self._models[model_id]

    def get_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration."""
        return self._models.get(model_id)

    def get_instance(self, model_id: str) -> Optional[ModelInference]:
        """Get or create model instance."""
        if model_id not in self._models:
            return None

        if model_id not in self._instances:
            config = self._models[model_id]
            self._instances[model_id] = create_model(config)

        return self._instances[model_id]

    def list_models(self, modality: str = "", model_type: ModelType = None) -> list[ModelConfig]:
        """List registered models."""
        models = list(self._models.values())

        if modality:
            models = [m for m in models if m.modality == modality]
        if model_type:
            models = [m for m in models if m.model_type == model_type]

        return models

    def clear(self):
        """Unload all models and clear registry."""
        for instance in self._instances.values():
            instance.unload()
        self._instances.clear()
        self._models.clear()


class AIService:
    """AI inference service for medical imaging."""

    def __init__(self, config: AIServiceConfig):
        self.config = config
        self.registry = ModelRegistry()
        self.status = ServiceStatus.OFFLINE
        self._validator = InputValidator()
        self._inference_log: list[InferenceLog] = []
        self._listeners: list[Callable[[str, Any], None]] = []

        # Register models from config
        for model_config in config.models:
            self.registry.register(model_config)

    def add_listener(self, callback: Callable[[str, Any], None]):
        """Add event listener."""
        self._listeners.append(callback)

    def _notify(self, event: str, data: Any = None):
        """Notify listeners."""
        for listener in self._listeners:
            try:
                listener(event, data)
            except Exception:
                pass

    def start(self):
        """Start the AI service."""
        self.status = ServiceStatus.STARTING
        self._notify("starting", None)

        try:
            if self.config.auto_load_models:
                self._load_all_models()
            self.status = ServiceStatus.READY
            self._notify("ready", None)
        except Exception as e:
            self.status = ServiceStatus.ERROR
            self._notify("error", str(e))

    def stop(self):
        """Stop the AI service."""
        self.registry.clear()
        self.status = ServiceStatus.OFFLINE
        self._notify("stopped", None)

    def _load_all_models(self):
        """Load all registered models."""
        for model_id in list(self.registry._models.keys()):
            instance = self.registry.get_instance(model_id)
            if instance and not instance.is_loaded():
                instance.load()

    def infer(
        self,
        model_id: str,
        image: np.ndarray,
        validate: bool = None,
    ) -> InferenceResult:
        """Run inference on an image."""
        import time
        import uuid

        if self.status != ServiceStatus.READY:
            raise RuntimeError(f"Service not ready: {self.status.value}")

        inference_id = str(uuid.uuid4())[:8]

        # Validate input
        should_validate = validate if validate is not None else self.config.validate_inputs
        if should_validate:
            is_valid, issues = self._validator.validate(image)
            if not is_valid:
                raise ValueError(f"Invalid input: {'; '.join(issues)}")

        # Get model
        model = self.registry.get_instance(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")

        # Load if needed
        if not model.is_loaded():
            model.load()

        self.status = ServiceStatus.BUSY
        self._notify("inference_started", {"model_id": model_id, "inference_id": inference_id})

        try:
            start_time = time.time()
            result = model.predict(image)
            inference_time = (time.time() - start_time) * 1000

            # Add quality score
            result.input_quality_score = self._validator.quality_score(image)

            # Log inference
            if self.config.log_inferences:
                self._log_inference(
                    inference_id=inference_id,
                    model_id=model_id,
                    input_shape=image.shape,
                    inference_time_ms=inference_time,
                    result=result,
                    success=True,
                )

            self.status = ServiceStatus.READY
            self._notify("inference_completed", {"model_id": model_id, "inference_id": inference_id})

            return result

        except Exception as e:
            self.status = ServiceStatus.READY
            if self.config.log_inferences:
                self._log_inference(
                    inference_id=inference_id,
                    model_id=model_id,
                    input_shape=image.shape,
                    inference_time_ms=0,
                    result=None,
                    success=False,
                    error=str(e),
                )
            raise

    def batch_infer(
        self,
        model_id: str,
        images: list[np.ndarray],
    ) -> list[InferenceResult]:
        """Run inference on a batch of images."""
        results = []
        for image in images[:self.config.max_batch_size]:
            results.append(self.infer(model_id, image))
        return results

    def _log_inference(
        self,
        inference_id: str,
        model_id: str,
        input_shape: tuple,
        inference_time_ms: float,
        result: Optional[InferenceResult],
        success: bool,
        error: str = "",
    ):
        """Log an inference."""
        log_entry = InferenceLog(
            inference_id=inference_id,
            model_id=model_id,
            timestamp=datetime.now(),
            input_shape=input_shape,
            inference_time_ms=inference_time_ms,
            result_summary=result.to_dict() if result else {},
            success=success,
            error_message=error,
        )
        self._inference_log.append(log_entry)

        # Keep log size bounded
        if len(self._inference_log) > 1000:
            self._inference_log = self._inference_log[-500:]

    def get_inference_log(self, limit: int = 100) -> list[InferenceLog]:
        """Get recent inference log."""
        return self._inference_log[-limit:]

    def get_statistics(self) -> dict:
        """Get service statistics."""
        total = len(self._inference_log)
        successful = sum(1 for log in self._inference_log if log.success)

        inference_times = [log.inference_time_ms for log in self._inference_log if log.success]
        avg_time = sum(inference_times) / len(inference_times) if inference_times else 0

        by_model = {}
        for log in self._inference_log:
            by_model[log.model_id] = by_model.get(log.model_id, 0) + 1

        return {
            "status": self.status.value,
            "total_inferences": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0,
            "avg_inference_time_ms": avg_time,
            "by_model": by_model,
            "registered_models": len(self.registry._models),
            "loaded_models": sum(1 for m in self.registry._instances.values() if m.is_loaded()),
        }


# Pre-configured models for medical imaging
DEFAULT_MODELS = [
    ModelConfig(
        model_id="us_classification_v1",
        name="Ultrasound Classification",
        model_type=ModelType.CLASSIFICATION,
        framework="onnx",
        modality="US",
        output_classes=["normal", "abnormal"],
        input_shape=(224, 224, 1),
        description="Classify ultrasound images as normal or abnormal",
    ),
    ModelConfig(
        model_id="xray_pneumonia_v1",
        name="Chest X-Ray Pneumonia Detection",
        model_type=ModelType.CLASSIFICATION,
        framework="onnx",
        modality="DX",
        body_part="CHEST",
        output_classes=["normal", "pneumonia"],
        input_shape=(224, 224, 1),
        description="Detect pneumonia in chest X-rays",
    ),
    ModelConfig(
        model_id="oct_segmentation_v1",
        name="OCT Retinal Layer Segmentation",
        model_type=ModelType.SEGMENTATION,
        framework="onnx",
        modality="OPT",
        body_part="EYE",
        output_classes=["background", "rnfl", "rpe", "drusen"],
        input_shape=(512, 512, 1),
        description="Segment retinal layers in OCT images",
    ),
    ModelConfig(
        model_id="thermal_roi_v1",
        name="Thermal ROI Detection",
        model_type=ModelType.DETECTION,
        framework="onnx",
        modality="TG",
        output_classes=["hot_spot", "cold_spot"],
        input_shape=(256, 256, 1),
        threshold=0.5,
        description="Detect thermal regions of interest",
    ),
]


def create_default_service(service_id: str = "imaging_ai") -> AIService:
    """Create an AI service with default models."""
    config = AIServiceConfig(
        service_id=service_id,
        name="Imaging AI Service",
        models=DEFAULT_MODELS,
    )
    return AIService(config)
