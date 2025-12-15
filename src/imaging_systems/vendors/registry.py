"""Vendor registry for managing and discovering vendor adapters."""

from typing import Optional, Type
from .base import VendorAdapter, DeviceCapabilities
from .ultrasound_vendors import (
    GEHealthcareUltrasound,
    PhilipsUltrasound,
    SiemensUltrasound,
    CanonUltrasound,
    SamsungUltrasound,
)
from .xray_vendors import (
    GEHealthcareXRay,
    PhilipsXRay,
    SiemensXRay,
    CareStreamXRay,
    FujifilmXRay,
)
from .oct_vendors import (
    ZeissOCT,
    HeidelbergOCT,
    TopconOCT,
    OptovueOCT,
)
from .thermal_vendors import (
    FLIRThermal,
    InfraTecThermal,
    TestoThermal,
)


class VendorRegistry:
    """Registry for managing vendor adapters."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._adapters: dict[str, Type[VendorAdapter]] = {}
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._register_default_adapters()
            self._initialized = True

    def _register_default_adapters(self):
        """Register all built-in vendor adapters."""
        # Ultrasound vendors
        self.register(GEHealthcareUltrasound)
        self.register(PhilipsUltrasound)
        self.register(SiemensUltrasound)
        self.register(CanonUltrasound)
        self.register(SamsungUltrasound)

        # X-ray vendors
        self.register(GEHealthcareXRay)
        self.register(PhilipsXRay)
        self.register(SiemensXRay)
        self.register(CareStreamXRay)
        self.register(FujifilmXRay)

        # OCT vendors
        self.register(ZeissOCT)
        self.register(HeidelbergOCT)
        self.register(TopconOCT)
        self.register(OptovueOCT)

        # Thermal vendors
        self.register(FLIRThermal)
        self.register(InfraTecThermal)
        self.register(TestoThermal)

    def register(self, adapter_class: Type[VendorAdapter]):
        """Register a vendor adapter class."""
        self._adapters[adapter_class.VENDOR_ID] = adapter_class

    def unregister(self, vendor_id: str):
        """Unregister a vendor adapter."""
        if vendor_id in self._adapters:
            del self._adapters[vendor_id]

    def get_adapter(
        self,
        vendor_id: str,
        model: str = "",
    ) -> Optional[VendorAdapter]:
        """Get an instance of a vendor adapter."""
        if vendor_id in self._adapters:
            return self._adapters[vendor_id](model)
        return None

    def get_adapter_class(self, vendor_id: str) -> Optional[Type[VendorAdapter]]:
        """Get the adapter class for a vendor."""
        return self._adapters.get(vendor_id)

    def list_vendors(self) -> list[dict]:
        """List all registered vendors."""
        vendors = []
        for vendor_id, adapter_class in self._adapters.items():
            vendors.append({
                "vendor_id": vendor_id,
                "vendor_name": adapter_class.VENDOR_NAME,
                "supported_models": adapter_class.SUPPORTED_MODELS,
            })
        return vendors

    def list_vendors_by_modality(self, modality: str) -> list[dict]:
        """List vendors that support a specific modality."""
        vendors = []
        for vendor_id, adapter_class in self._adapters.items():
            adapter = adapter_class()
            capabilities = adapter.get_capabilities()
            if modality in capabilities.modalities:
                vendors.append({
                    "vendor_id": vendor_id,
                    "vendor_name": adapter_class.VENDOR_NAME,
                    "supported_models": adapter_class.SUPPORTED_MODELS,
                })
        return vendors

    def find_adapter_for_model(self, model_name: str) -> Optional[str]:
        """Find the vendor ID for a specific model."""
        for vendor_id, adapter_class in self._adapters.items():
            if model_name in adapter_class.SUPPORTED_MODELS:
                return vendor_id
        return None

    def get_all_supported_models(self) -> dict[str, list[str]]:
        """Get all supported models grouped by vendor."""
        models = {}
        for vendor_id, adapter_class in self._adapters.items():
            models[adapter_class.VENDOR_NAME] = adapter_class.SUPPORTED_MODELS
        return models


# Global registry instance
_registry = VendorRegistry()


def get_vendor_adapter(vendor_id: str, model: str = "") -> Optional[VendorAdapter]:
    """Get a vendor adapter instance."""
    return _registry.get_adapter(vendor_id, model)


def list_all_vendors() -> list[dict]:
    """List all registered vendors."""
    return _registry.list_vendors()


def list_vendors_for_modality(modality: str) -> list[dict]:
    """List vendors for a specific modality."""
    return _registry.list_vendors_by_modality(modality)


def register_custom_adapter(adapter_class: Type[VendorAdapter]):
    """Register a custom vendor adapter."""
    _registry.register(adapter_class)


def find_vendor_for_model(model_name: str) -> Optional[str]:
    """Find vendor ID for a model name."""
    return _registry.find_adapter_for_model(model_name)
