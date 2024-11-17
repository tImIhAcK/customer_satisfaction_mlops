from dataclasses import dataclass

@dataclass
class ModelNameConfig:
    """Model configuration"""
    model_name: str = "LinearRegression"