import torch
import json
import os

from milp.backend import configure_backend, get_backend_status

from configure.advanced import BaseSettings, AdvancedSettings

class GlobalSettings(BaseSettings):

    def __init__(self):
        # data precision
        torch.set_default_dtype(torch.float32)
        
        # restart
        self.use_restart = True
        
        # stabilize
        self.use_mip_tightening = True
        
        # attack
        self.use_attack = True
        self.attack_interval = 10
        
        # mip verify
        self.use_mip_verify = True 
        self.mip_backend = 'auto'
        self.mip_backend_available = False
        self.mip_backend_name = None
        
        # threshold for input/hidden splitting: 
        self.input_splitting_threshold = 0.5  # > 0.5: use input splitting
        self.hidden_splitting_threshold = 0.01  # < 0.01: use hidden splitting
        self.safety_num_input_perturbed = 200 # < 200: use input splitting
        
        # preprocess
        self.skip_preprocess = False
        
        # proof
        self.use_save_reasoning_step = False
        
        # early stopping
        self.max_iterations = 1e9
        self.skip_initial_worst_bound = -1e6
        self.max_domains = 1e9
        
        # decomposition
        self.use_decompose = False
        
    
    def _add_advanced_settings(self, args=None):
        advanced = AdvancedSettings(args)
        for key, value in advanced.__dict__.items():
            if not key.startswith('_') and key != 'advanced_settings':
                setattr(self, key, value)

    def _configure_and_apply_mip_backend(self, args):
        if args is not None and hasattr(args, 'mip_backend') and args.mip_backend is not None:
            self.mip_backend = args.mip_backend

        try:
            configure_backend(self.mip_backend)
        except ValueError:
            print(f"[!] Unsupported mip_backend={self.mip_backend!r}. Falling back to 'auto'.")
            self.mip_backend = 'auto'
            configure_backend(self.mip_backend)
        backend_status = get_backend_status()
        self.mip_backend_available = backend_status['available']
        self.mip_backend_name = backend_status['active']
        if not self.mip_backend_available:
            detail = f" ({backend_status['error']})" if backend_status['error'] else ""
            print(
                "[!] MIP backend unavailable for "
                f"{self.mip_backend!r}. MIP-based features will be disabled.{detail}"
            )

        self.use_mip_verify = self.use_mip_verify and self.mip_backend_available
        self.use_mip_tightening = self.use_mip_tightening and self.mip_backend_available
        if hasattr(self, 'use_mip_attack'):
            self.use_mip_attack = self.use_mip_attack and self.mip_backend_available
    
    def setup(self, args=None):
        # add advanced settings
        self._add_advanced_settings(args)

        self._configure_and_apply_mip_backend(args)

        if args is not None:
            if hasattr(args, 'disable_attack'):
                self.use_attack = args.disable_attack
            if hasattr(args, 'disable_restart'):
                self.use_restart = args.disable_restart
            if hasattr(args, 'disable_stabilize'):
                self.use_mip_tightening = args.disable_stabilize and self.mip_backend_available

        # load specific settings from json
        if args is not None and args.setting_file is not None:
            assert os.path.exists(args.setting_file), f"Setting file not found: {args.setting_file=}"
            settings = json.load(open(args.setting_file))
            for key, value in settings.items():
                assert hasattr(self, key), f"Unknown setting: {key=}"
                setattr(self, key, value)

        if self.use_save_reasoning_step:
            torch.set_default_dtype(torch.float64)
            print(f'[!] Using float64 for proof generation')
            
Settings = GlobalSettings()
