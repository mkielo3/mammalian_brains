from abc import ABC, abstractmethod
import torch
import os

class ModalityTrainer(ABC):
   """
   Define functions common to all modalities
   """

   @abstractmethod
   def setup_model(self):
       """Download or train necessary models and data"""
       pass

   @abstractmethod 
   def setup_som(self):
       """Load local data and models necessary for SOM"""
       pass
       
   @abstractmethod
   def generate_static(self):
       """Generate patch of static"""
       pass

   @abstractmethod
   def calculate_activations(self):
       """Get activations for patch"""
       pass

   @abstractmethod
   def get_patches(self):
       """Retur list of patches to probe"""
       pass
   
   def initialize_som(self, SOM):
       return SOM(som_size=self.som_size, is_torus=False, balance=0.5)