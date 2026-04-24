import torch
import numpy as np

from model import BinaryConv2d


def save_1bit_model(model, filepath):
    state_dict = model.state_dict()
    packed_dict = {}

    binary_layers = [name for name, mod in model.named_modules() if isinstance(mod, BinaryConv2d)]

    for key, tensor in state_dict.items():
        is_binary_weight = any(key == f"{b_name}.weight" for b_name in binary_layers)

        if is_binary_weight:
            w = tensor.detach()
            
            # Calculate the XNOR scale factor
            scale = w.abs().mean(dim=(1, 2, 3), keepdim=True)
            
            # Binarize to bool
            w_bool = (w >= 0).cpu().numpy().astype(np.uint8)
            
            # Pack 8 booleans into a single byte
            w_packed = np.packbits(w_bool.reshape(-1))

            # Save the compressed bits, the scale, and the original shape
            packed_dict[key + '_packed'] = w_packed
            packed_dict[key + '_scale'] = scale.cpu()
            packed_dict[key + '_shape'] = w.shape
        else:
            packed_dict[key] = tensor.cpu()

    torch.save(packed_dict, filepath)
    print(f"✅ 1-Bit Model saved to {filepath}")
