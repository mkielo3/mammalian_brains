import pickle
import os

def save_output_to_pickle(output, experiment_name):
   modality_name = output["modality"]
   filename = f"{modality_name}.pkl"
   base_dir = f"output/{experiment_name}"
   os.makedirs(base_dir, exist_ok=True)
   filepath = os.path.join(base_dir, filename)
   with open(filepath, 'wb') as f:
       pickle.dump(output, f)
   print(f"Saved output to: {filepath}")
   return filepath

def load_pickle_output(experiment_name, modality):
   filename = f"output/{experiment_name}/{modality}.pkl"
   with open(filename, 'rb') as f:
       output = pickle.load(f)
   return output


def save_som_plot(experiment_name, modality_list, args):
   from analysis.som_smoothness import gen_and_save_som_charts
   charts = gen_and_save_som_charts(modality_list, args, load_pickle_output)
   charts.save(f"output/{experiment_name}/som_charts.png")
   return charts


def save_rf_plot(experiment_name, modality_list):
   from analysis.receptive_field import plot_rf
   charts = plot_rf(experiment_name, modality_list, load_pickle_output)
   charts.save(f"output/{experiment_name}/rf_charts.png")
   return charts