import torch
import torch.nn as nn
import yaml
from models import Deformer

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def init_model(config):
    """Initializes the Deformer model based on config."""
    if config["model"] == 'Deformer':
        return Deformer(
            num_chan=config["num_chan"],
            num_time=config["num_time"],
            temporal_kernel=config["kernel_length"],
            num_kernel=config["T"],
            num_classes=config["num_class"],
            depth=int(config["num_layers"] - 2),
            heads=config["AT"],
            mlp_dim=config["AT"],
            dim_head=config["AT"],
            dropout=config["dropout"]
        )
    return None

class DLModel(nn.Module):
    def __init__(self, config):
        super(DLModel, self).__init__()
        self.config = config
        self.net = init_model(self.config)

        # Hook for capturing latent output
        self.activations = {}
        self.register_latent_hook()

    def register_latent_hook(self):
        """Register forward hook to capture latent features from the MLP head."""
        layer_name = "mlp_head.0"  # Adjust this layer name if needed

        for name, layer in self.net.named_modules():
            if name == layer_name:
                layer.register_forward_hook(self.hook_fn)
                print(f"✅ Latent Hook Registered on Layer: {name}")
                return  
        print(f"⚠️ Warning: Layer '{layer_name}' not found!")

    def hook_fn(self, module, input, output):
        """Store the output of the hooked layer."""
        self.activations["latent"] = input

    def forward(self, x):
        self.activations.clear()  # Reset stored activations
        return self.net(x)

    def latent_output(self):
        """Retrieve latent feature output."""
        if "latent" in self.activations:
            return self.activations["latent"]
        else:
            print("⚠️ Warning: No latent output stored. Run a forward pass first!")
            return None


if __name__ == "__main__":
    config_path = "configs/config-deformer.yaml"
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    # Initialize model
    model = DLModel(config)

    # Load checkpoint using torch.load()
    checkpoint_path = "save/logs_IMGCLF_Deformer/sub6_subject_clf_16th_feb/checkpoints/epoch=337-step=58136.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cuda")

    # Load only the model weights
    model.net.load_state_dict(checkpoint['state_dict'], strict=False)
    print("✅ Model checkpoint loaded successfully!")

    # Move model to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Register hook manually (just in case)
    model.register_latent_hook()

    # Print model layers
    for name, layer in model.net.named_modules():
        print(f"Layer: {name} -> {layer}")

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Perform inference on a sample input
        sample_input = torch.randn(16, 128, 440).to(device)  # Ensure input is on the correct device
        output = model(sample_input)
        activations = model.activations

        print(f"Output Shape: {output.shape}")
        if "latent" in activations:
            print(f"Latent Output Shape: {activations['latent'][0].shape}")
        else:
            print("⚠️ Warning: No latent output captured.")
