# src/retrain.py
from src.model import retrain_model_from_upload

def retrain_with_new_folder(upload_path, epochs=5):
    """
    Wrapper function that calls the actual retrain function in src/model.py
    
    Args:
        upload_path: Path to the uploaded and extracted folder containing class subfolders
        epochs: Number of epochs to train for
    
    Returns:
        model: The retrained model
        history: Training history object
    """
    # Call the actual retrain function with correct parameter name
    model, history, metrics = retrain_model_from_upload(
        upload_folder=upload_path,  # Note: parameter name is 'upload_folder', not 'upload_path'
        epochs=epochs
    )
    
    return model, history