import models


def call(args):
    """
    Dispatches the call to the appropriate model implementation.
    
    Args:
        args: Command line arguments containing model choice and other parameters
    
    Returns:
        Result from the selected model
    
    Raises:
        Exception: If the requested model is not implemented
    """
    model = args.model
    
    # Map of supported models to their implementation functions
    model_map = {
        "starcoder": models.starcoder,
        "gpt3": models.gpt3,
        "gpt4": models.gpt4,
        "mistral": models.mistral,
        "deepseek": models.deepseek,
        "deepseek_small": models.deepseek_local,
        "deepseek_large": models.deepseek_local,
        "deepseek_local": models.deepseek_local, 
        "gpt4o": models.gpt4o,
        "test": models.test if hasattr(models, "test") else None
    }
        
    if model not in model_map or model_map[model] is None:
        # Display available models
        available_models = [k for k, v in model_map.items() if v is not None]
        raise Exception(f"Model '{model}' is not implemented. Available models: {', '.join(available_models)}")
    return model_map[model](args)    
    
    raise Exception(f"Model '{model}' is not implemented. Available models: {', '.join(model_map.keys())}")
