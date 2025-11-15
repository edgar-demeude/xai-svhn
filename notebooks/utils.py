import os
import torch

def save_checkpoint(model, path="../models/model_checkpoint.pth"):
    """
    Saves only the internal state (weights) of the model.
    """
    # Create the folder if it does not exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save(model.state_dict(), path)
    print(f"Model saved in : {path}")

def load_checkpoint(model, path):
    """
    Loads the saved weights into a new model instance.
    """
    if not torch.cuda.is_available():
        map_location = torch.device('cpu')
    else:
        map_location = None
        
    model.load_state_dict(torch.load(path, map_location=map_location))
    print(f"Model loaded from : {path}")
    return model