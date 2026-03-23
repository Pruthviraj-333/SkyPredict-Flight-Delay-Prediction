
import sys
import os
import pandas as pd

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.model_service import ModelService

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(root_dir, "models")
    data_dir = os.path.join(root_dir, "data", "processed")
    
    print("--- Initializing ModelService ---")
    service = ModelService(models_dir=models_dir, data_dir=data_dir)
    
    print(f"Fallback Regressor Loaded: {service.fallback_reg_model is not None}")
    print(f"Primary Regressor Loaded: {service.primary_reg_model is not None}")
    
    if service.fallback_reg_model is None:
        print("ERROR: Fallback regressor NOT loaded!")
        return

    print("\n--- Testing Prediction with Fallback Regressor ---")
    # AA100: JFK -> LAX
    result = service.predict(
        carrier="AA",
        origin="JFK",
        dest="LAX",
        date_str="2025-10-25",
        dep_time="1030"
    )
    
    res_dict = result.to_dict()
    print(f"Is Delayed: {res_dict['is_delayed']}")
    print(f"Delay Prob: {res_dict['delay_probability']:.4f}")
    print(f"Predicted Delay Minutes: {res_dict['predicted_delay_minutes']} min")
    print(f"Model Used: {res_dict['model_used']}")
    
    if "predicted_delay_minutes" in res_dict:
        print("SUCCESS: predicted_delay_minutes field found in output.")
    else:
        print("FAILURE: predicted_delay_minutes field MISSING from output.")

if __name__ == "__main__":
    main()
