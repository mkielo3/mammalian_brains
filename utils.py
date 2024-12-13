import pickle
import datetime
import os

def save_output_to_pickle(output, experiment_name):
   timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
   modality_name = output["modality"]
   filename = f"{modality_name}_{timestamp}.pkl"
   base_dir = f"output/{experiment_name}"
   os.makedirs(base_dir, exist_ok=True)
   filepath = os.path.join(base_dir, filename)
   
   with open(filepath, 'wb') as f:
       pickle.dump(output, f)
   
   print(f"Saved output to: {filepath}")
   return filepath
