"""Closed codes of the GPU presets and of the GPU sessions of the product (MFL023).

They are not members of the engine's enum (`engine/codes.py`): `provider_not_idle` is already a code of the bench
alone (`bench/codes.py`), and the bench enum refuses a duplicate value. Every refusal of the presets and of the
sessions carries one of these codes and, at most, a local detail without values read from a provider. Standard
library only: the API process imports this module through `gpu.presets`.
"""

GPU_PRESET_UNKNOWN = "gpu_preset_unknown"            # no preset of that name in the versioned file
GPU_PRESET_NOT_ENABLED = "gpu_preset_not_enabled"    # a preset this instance does not enable
GPU_PRESET_SUPERSEDED = "gpu_preset_superseded"      # listed for its measures, never armed
GPU_VOLUME_UNSUPPORTED = "gpu_volume_unsupported"    # a global volume: arrives with run 4 of the GPU sprint
GPU_PRESET_INVALID = "gpu_preset_invalid"            # the versioned file outside its closed schema
PROVIDER_NOT_IDLE = "provider_not_idle"              # the provider has a live window of another session
ENDPOINT_OUT_OF_PINS = "endpoint_out_of_pins"        # the endpoint differs from its pins or from its preset
MODAL_NO_CAPACITY = "modal_no_capacity"              # Modal had no GPU for the session
SESSION_STOPPED = "session_stopped"                  # the session could not see its zero: the provider is stopped
GPU_UNRECOVERABLE = "gpu_unrecoverable"              # a window left by a dead worker cannot be closed

CODES = (GPU_PRESET_UNKNOWN, GPU_PRESET_NOT_ENABLED, GPU_PRESET_SUPERSEDED, GPU_VOLUME_UNSUPPORTED, GPU_PRESET_INVALID,
         PROVIDER_NOT_IDLE, ENDPOINT_OUT_OF_PINS, MODAL_NO_CAPACITY, SESSION_STOPPED, GPU_UNRECOVERABLE)


class GpuError(Exception):
    """A refusal with a closed code of this module and an optional local detail (never a value from a provider)."""

    def __init__(self, code, detail=None):
        if code not in CODES:
            raise ValueError("not a GPU code: %r" % (code,))
        super().__init__(code if detail is None else "%s: %s" % (code, detail))
        self.code, self.detail = code, detail
