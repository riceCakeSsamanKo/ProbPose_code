import argparse
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from thop import profile

# mmpose imports
from mmpose.models import build_pose_estimator

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate model FLOPs and Parameters')
    parser.add_argument('config', help='test config file path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmpose'))

    # Build the model
    model = build_pose_estimator(cfg.model)
    model.eval()

    # Get input size from config
    # In mmpose configs, the input size is typically in (W, H) format.
    # The model expects the input tensor in (N, C, H, W) format.
    # From your config log, input_size=(192, 256), so W=192, H=256.
    try:
        # Find the 'TopdownAffine' transform to get the input size
        input_size = None
        for transform in cfg.test_dataloader.dataset.pipeline:
            if transform.get('type') == 'TopdownAffine':
                input_size = transform['input_size']
                break
        if input_size is None:
            raise ValueError("Could not find 'TopdownAffine' in the pipeline to determine input size.")
            
        dummy_input = torch.randn(1, 3, input_size[1], input_size[0])
    except Exception as e:
        print(f"Failed to determine input size from config: {e}")
        print("Using default size (1, 3, 256, 192). Please check if this is correct.")
        dummy_input = torch.randn(1, 3, 256, 192)


    # Calculate FLOPs and Params
    # The model's forward method requires both 'inputs' and 'data_samples'.
    # We pass None for data_samples as it's not used in 'tensor' mode.
    flops, params = profile(model, inputs=(dummy_input, None), verbose=False)

    # Convert to GFLOPs and MParams for readability
    gflops = flops / 1e9
    mparams = params / 1e6

    print(f"\nModel: \t\t{model.__class__.__name__}")
    print(f"Input shape: \t{list(dummy_input.shape)}")
    print(f"FLOPs: \t\t{flops:,.0f}")
    print(f"GFLOPs: \t\t{gflops:.4f} G")
    print(f"Params: \t\t{params:,.0f}")
    print(f"MParams: \t\t{mparams:.4f} M")


if __name__ == '__main__':
    main()