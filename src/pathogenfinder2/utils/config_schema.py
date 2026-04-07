"""
JSON Schema definitions for PathogenFinder2 configuration validation.

:data:`PREDICTION_SCHEMA` covers the base prediction config (loaded from
``config_base.json`` with weights appended).  :data:`USER_CONFIG_SCHEMA`
covers user-supplied training/fine-tuning configs.
"""

PREDICTION_SCHEMA = {
    "type": "object",
    "required": ["Misc Parameters", "Model Parameters"],
    "additionalProperties": True,
    "properties": {
        "Misc Parameters": {
            "type": "object",
            "required": ["Report Results"],
            "properties": {
                "Report Results": {"type": "string", "enum": ["file", "wandb"]},
                "Results Folder": {"type": "string"},
                "Actions": {"type": "string"},
            },
        },
        "Model Parameters": {
            "type": "object",
            "required": ["Model Name", "Network Weights", "Batch Size",
                         "Mixed Precision", "Loss Function"],
            "properties": {
                "Model Name": {
                    "type": "string",
                    "enum": ["ConvNext-AddAtt", "ConvNext", "Conv1D-AddAtt",
                             "Conv1D", "FNN", "AddAtt", "DenseNet", "DenseNet-AddAtt"],
                },
                "Network Weights": {"type": "array"},
                "Batch Size": {"type": "integer", "minimum": 1},
                "Mixed Precision": {"type": "boolean"},
                "Loss Function": {"type": "string", "enum": ["bcelogits", "bce"]},
                "Input Dimensions": {"type": "integer", "minimum": 1},
                "Out Dimensions": {"type": "integer", "minimum": 1},
                "Memory Report": {"type": "boolean"},
            },
        },
    },
}

USER_CONFIG_SCHEMA = {
    "type": "object",
    "required": ["Misc Parameters", "Model Parameters", "Train Parameters"],
    "additionalProperties": True,
    "properties": {
        "Misc Parameters": PREDICTION_SCHEMA["properties"]["Misc Parameters"],
        "Model Parameters": PREDICTION_SCHEMA["properties"]["Model Parameters"],
        "Train Parameters": {
            "type": "object",
            "required": ["Optimizer", "Learning Rate", "Weight Decay",
                         "Epochs", "Save Model"],
            "properties": {
                "Optimizer": {},  # str before conversion, class after — allow any
                "Learning Rate": {"type": "number", "exclusiveMinimum": 0},
                "Weight Decay": {"type": "number", "minimum": 0},
                "Epochs": {"type": "integer", "minimum": 1},
                "Save Model": {
                    "type": "string",
                    "enum": ["last_epoch", "early_stopping", "best_epoch"],
                },
                "Lr Scheduler": {},
                "Warm Up": {},
            },
        },
    },
}
