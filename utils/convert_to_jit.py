import torch
import timm

def convert_model():
    # Load the saved model
    saved_model = torch.load('enet_b2_8_best.pt', map_location=torch.device('cpu'))
    
    # Set to eval mode for inference
    saved_model.eval()
    
    # Create an example input tensor
    example_input = torch.randn(1, 3, 224, 224)  # EfficientNet-B0 typically uses 224x224
    
    # Convert to TorchScript using tracing
    scripted_model = torch.jit.script(saved_model)
    
    # Save the TorchScript model
    scripted_model.save('enet_b2_8_best_jit.pt')
    print("Model converted successfully!")

if __name__ == '__main__':
    convert_model()