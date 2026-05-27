# -*- coding: utf-8 -*-
"""
Model Export Utilities
=======================
Export trained models to multiple formats for deployment:
  1. TorchScript (.pt) — Already implemented in train.py
  2. ONNX (.onnx) — Universal format for cross-platform inference
  3. Standalone Python module — Self-contained inference without project dependencies
"""
import os
import sys
from pathlib import Path
import json
import logging

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import cfg
from neural_engine.model import MultiHeadTradingModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_to_onnx(
    model_path: str,
    output_path: str,
    input_dim: int,
    opset_version: int = 14,
) -> None:
    """
    Export PyTorch model to ONNX format for deployment on non-Python platforms.

    ONNX (Open Neural Network Exchange) is an open format supported by:
    - TensorFlow, PyTorch, scikit-learn
    - C++, C#, Java inference engines (ONNX Runtime)
    - Mobile (iOS, Android via Core ML, TensorFlow Lite)
    - Web (ONNX.js for browser inference)

    Args:
        model_path: Path to trained .pth model weights
        output_path: Where to save .onnx file
        input_dim: Number of input features (must match training)
        opset_version: ONNX opset version (14 is widely supported)
    """
    try:
        import onnx
        import onnxruntime
    except ImportError:
        logger.error("ONNX export requires: pip install onnx onnxruntime")
        return

    device = "cpu"  # Always export to CPU for compatibility
    model = MultiHeadTradingModel(input_dim=input_dim).to(device)

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)
    model.eval()

    # Create example input (batch_size=1, window_size, num_features)
    dummy_input = torch.zeros(1, cfg.model.WINDOW_SIZE, input_dim, dtype=torch.float32)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["window_features"],
        output_names=["direction", "sizing", "magnitude", "time"],
        dynamic_axes={
            "window_features": {0: "batch_size"},
            "direction": {0: "batch_size"},
            "sizing": {0: "batch_size"},
            "magnitude": {0: "batch_size"},
            "time": {0: "batch_size"},
        },
    )

    # Verify the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Test inference with ONNX Runtime
    session = onnxruntime.InferenceSession(output_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: dummy_input.numpy()})

    logger.info(f"✅ ONNX export successful: {output_path}")
    logger.info(f"   Model size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    logger.info(f"   Input shape: (batch_size, {cfg.model.WINDOW_SIZE}, {input_dim})")
    logger.info(f"   Output shapes: {[out.shape for out in outputs]}")

    # Save metadata
    metadata = {
        "model_type": "MultiHeadTradingModel",
        "window_size": cfg.model.WINDOW_SIZE,
        "input_dim": input_dim,
        "hidden_dim": cfg.model.HIDDEN_DIM,
        "num_layers": cfg.model.NUM_LAYERS,
        "num_heads": cfg.model.NUM_HEADS,
        "interval": cfg.model.INTERVAL,
        "symbols": cfg.model.SYMBOLS,
        "max_atr_target_pct": cfg.testing.MAX_ATR_TARGET_PCT,
        "max_atr_stop_pct": cfg.testing.MAX_ATR_STOP_PCT,
        "lookahead_bars": cfg.training.LOOKAHEAD_BARS,
        "onnx_opset_version": opset_version,
    }

    metadata_path = output_path.replace(".onnx", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✅ Metadata saved: {metadata_path}")


def export_to_tf_savedmodel(
    model_path: str,
    output_dir: str,
    input_dim: int,
) -> None:
    """
    Export PyTorch model to TensorFlow SavedModel format.

    Useful for:
    - TensorFlow Serving (production deployment)
    - TensorFlow Lite (mobile/embedded)
    - Cloud ML platforms (GCP AI Platform, AWS SageMaker)

    Args:
        model_path: Path to trained .pth model weights
        output_dir: Directory to save TF SavedModel
        input_dim: Number of input features
    """
    try:
        import tf2onnx
        import tensorflow as tf
    except ImportError:
        logger.error("TF export requires: pip install onnx tf2onnx tensorflow")
        return

    # First export to ONNX
    temp_onnx = "temp_model.onnx"
    export_to_onnx(model_path, temp_onnx, input_dim)

    # Convert ONNX to TensorFlow SavedModel
    os.system(f"python -m tf2onnx.convert --onnx {temp_onnx} --output {output_dir} --saved-model")

    # Cleanup
    if os.path.exists(temp_onnx):
        os.remove(temp_onnx)

    logger.info(f"✅ TensorFlow SavedModel exported: {output_dir}")


def create_standalone_inference_module(
    model_path: str,
    scaler_mean_path: str,
    scaler_scale_path: str,
    output_path: str,
    input_dim: int,
) -> None:
    """
    Create a completely self-contained inference script.

    This generates a single .py file that:
    - Loads the TorchScript model
    - Includes scaler parameters as embedded constants
    - Has no dependencies on your project structure
    - Can be copied anywhere and run standalone

    Perfect for:
    - Sharing with others (doesn't require your full codebase)
    - Deployment to servers (just copy one file)
    - Trading bots (minimal dependencies)

    Args:
        model_path: Path to trained .pth model
        scaler_mean_path: Path to scaler_mean.npy
        scaler_scale_path: Path to scaler_scale.npy
        output_path: Where to save the standalone .py file
        input_dim: Number of input features
    """
    # Load scaler parameters
    mean = np.load(scaler_mean_path)
    scale = np.load(scaler_scale_path)

    # Generate standalone Python code
    standalone_code = f'''#!/usr/bin/env python3
"""
Standalone Trading Model Inference
====================================
Auto-generated from: {model_path}

This file is completely self-contained. To use:

    python {os.path.basename(output_path)} --input data.csv

Requirements: torch, numpy, pandas (no custom dependencies)
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# EMBEDDED MODEL CONFIGURATION (frozen at training time)
# ═══════════════════════════════════════════════════════════════════

WINDOW_SIZE = {cfg.model.WINDOW_SIZE}
INPUT_DIM = {input_dim}
MAX_ATR_TARGET_PCT = {cfg.testing.MAX_ATR_TARGET_PCT}
MAX_ATR_STOP_PCT = {cfg.testing.MAX_ATR_STOP_PCT}
LOOKAHEAD_BARS = {cfg.training.LOOKAHEAD_BARS}
SIGNAL_MARGIN_THRESHOLD = {cfg.testing.SIGNAL_MARGIN_THRESHOLD}
AI_CONFIDENCE_THRESHOLD = {cfg.testing.AI_CONFIDENCE_THRESHOLD}

# Scaler parameters (fitted on training data)
SCALER_MEAN = np.array({mean.tolist()}, dtype=np.float32)
SCALER_SCALE = np.array({scale.tolist()}, dtype=np.float32)

# ═══════════════════════════════════════════════════════════════════
# INFERENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def load_model(model_path: Path) -> torch.nn.Module:
    """Load the TorchScript model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model


@torch.no_grad()
def predict(model: torch.nn.Module, features: np.ndarray) -> dict:
    """
    Run inference on a feature window.

    Args:
        model: Loaded TorchScript model
        features: (window_size, input_dim) array of features

    Returns:
        dict with: signal, confidence, long/neutral/short probs, tp_pct, sl_pct, etc.
    """
    if features.shape != (WINDOW_SIZE, INPUT_DIM):
        raise ValueError(f"Expected shape ({{WINDOW_SIZE}}, {{INPUT_DIM}}), got {{features.shape}}")

    # Scale features
    features_scaled = (features - SCALER_MEAN) / (SCALER_SCALE + 1e-8)

    # Convert to tensor
    x = torch.from_numpy(features_scaled).float().unsqueeze(0)
    device = next(model.parameters()).device
    x = x.to(device)

    # Forward pass
    direction, sizing, magnitude, time_out = model(x)

    # Decode outputs
    probs = torch.softmax(direction, dim=1).cpu().numpy()[0]
    sizing_vals = sizing.cpu().numpy()[0]
    magnitude_val = float(magnitude.cpu().numpy()[0][0])
    time_val = float(time_out.cpu().numpy()[0][0])

    long_p, neutral_p, short_p = [float(v) for v in probs]

    # Apply signal logic
    verdict = "NEUTRAL"
    confidence = neutral_p

    if long_p - neutral_p >= SIGNAL_MARGIN_THRESHOLD and long_p >= AI_CONFIDENCE_THRESHOLD:
        verdict, confidence = "LONG", long_p
    elif short_p - neutral_p >= SIGNAL_MARGIN_THRESHOLD and short_p >= AI_CONFIDENCE_THRESHOLD:
        verdict, confidence = "SHORT", short_p

    return {{
        "signal": verdict,
        "confidence": confidence,
        "long": long_p,
        "neutral": neutral_p,
        "short": short_p,
        "take_profit_pct": float(sizing_vals[1]) * MAX_ATR_TARGET_PCT,
        "stop_loss_pct": float(sizing_vals[2]) * MAX_ATR_STOP_PCT,
        "qty_ratio": float(sizing_vals[0]),
        "expected_move_pct": magnitude_val * MAX_ATR_TARGET_PCT,
        "bars_to_target": max(1, round(time_val * LOOKAHEAD_BARS)),
    }}


def batch_predict(model: torch.nn.Module, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run inference on a DataFrame of features.

    Args:
        model: Loaded model
        features_df: DataFrame with {INPUT_DIM} feature columns, indexed by time

    Returns:
        DataFrame with predictions added
    """
    if len(features_df) < WINDOW_SIZE:
        raise ValueError(f"Need at least {{WINDOW_SIZE}} rows, got {{len(features_df)}}")

    predictions = []

    for i in range(WINDOW_SIZE, len(features_df) + 1):
        window = features_df.iloc[i - WINDOW_SIZE : i].values
        pred = predict(model, window)
        pred["timestamp"] = features_df.index[i - 1]
        predictions.append(pred)

    return pd.DataFrame(predictions).set_index("timestamp")


# ═══════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Standalone trading model inference")
    parser.add_argument("--model", required=True, help="Path to TorchScript .pt model")
    parser.add_argument("--input", required=True, help="Path to CSV with features")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {{args.model}}...")
    model = load_model(Path(args.model))

    # Load features
    print(f"Loading features from {{args.input}}...")
    features_df = pd.read_csv(args.input, index_col=0, parse_dates=True)

    if features_df.shape[1] != INPUT_DIM:
        raise ValueError(f"Expected {{INPUT_DIM}} features, got {{features_df.shape[1]}}")

    # Run inference
    print("Running inference...")
    predictions = batch_predict(model, features_df)

    # Save results
    predictions.to_csv(args.output)
    print(f"✅ Predictions saved to {{args.output}}")
    print(f"   Rows: {{len(predictions)}}")
    print(f"   Signals: {{predictions['signal'].value_counts().to_dict()}}")


if __name__ == "__main__":
    main()
'''

    # Write to file
    with open(output_path, "w") as f:
        f.write(standalone_code)

    # Make executable
    os.chmod(output_path, 0o755)

    logger.info(f"✅ Standalone inference module created: {output_path}")
    logger.info(f"   Usage: python {output_path} --model model.pt --input features.csv")


def export_all_formats(model_dir: str = "models", input_dim: int = None) -> None:
    """
    Export model to all supported formats.

    Creates:
    - model.onnx (ONNX format)
    - model_metadata.json (config info)
    - standalone_inference.py (self-contained script)
    - tf_model/ (TensorFlow SavedModel) [optional]
    """
    model_path = os.path.join(model_dir, "trading_model.pth")
    scaler_mean = os.path.join(model_dir, "scaler_mean.npy")
    scaler_scale = os.path.join(model_dir, "scaler_scale.npy")

    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        logger.error("Run training first: python neural_engine/train.py")
        return

    if input_dim is None:
        # Infer from scaler
        input_dim = len(np.load(scaler_mean))

    logger.info(f"Exporting model with {input_dim} input features...")

    # 1. ONNX Export
    onnx_path = os.path.join(model_dir, "trading_model.onnx")
    export_to_onnx(model_path, onnx_path, input_dim)

    # 2. Standalone Python Module
    standalone_path = os.path.join(model_dir, "standalone_inference.py")
    create_standalone_inference_module(
        model_path, scaler_mean, scaler_scale, standalone_path, input_dim
    )

    # 3. TensorFlow (optional — commented out by default, uncomment if needed)
    # tf_dir = os.path.join(model_dir, "tf_model")
    # export_to_tf_savedmodel(model_path, tf_dir, input_dim)

    logger.info("=" * 60)
    logger.info("✅ ALL EXPORTS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"📦 TorchScript:  {model_dir}/trading_model_scripted.pt (already exists)")
    logger.info(f"📦 ONNX:         {onnx_path}")
    logger.info(f"📦 Standalone:   {standalone_path}")
    logger.info(f"📦 Metadata:     {onnx_path.replace('.onnx', '_metadata.json')}")
    logger.info("")
    logger.info("USAGE:")
    logger.info(f"  • Python:  python {standalone_path} --model models/trading_model_scripted.pt --input data.csv")
    logger.info(f"  • ONNX:    Use onnxruntime.InferenceSession('{onnx_path}')")
    logger.info(f"  • C++/C#:  Load {onnx_path} with ONNX Runtime")


if __name__ == "__main__":
    from neural_engine.feature_utils import get_feature_columns
    input_dim = len(get_feature_columns())
    export_all_formats(input_dim=input_dim)
