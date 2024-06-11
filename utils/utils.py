import numpy as np
import shutil
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import loadmat
from pathlib import Path
from datetime import date, datetime
from torch.utils.tensorboard import SummaryWriter
import json
from functools import partial

try:
    import wandb
    wandb_installed = True
except ImportError:
    wandb_installed = False

def print_divider(l=100):
    print("-"*l)

class ModelAnalyzer:
    def __init__(self, model):
        self.model = model
        self.rough_total_trainable = 0

    def print_model_arch(self):
        print_divider()
        print("[Info]: Model Definition")
        print(self.model)
        print_divider()

    def report_layerwise_parameters(self):
        self.rough_total_trainable = 0 # for simplicity, we ignore all the normalization layers, so this number is slightly off

        print(f"\n[Info]: Reporting layerwise trainable #param by thier param name\n{self.model.__class__.__name__}")
        for name, param in self.model.named_parameters():
            if param.requires_grad is True:
                if 'bn' in name or 'ln' in name or 'bias' in name:
                    #note: bias are still trainable, just skipping them for ease of estimation
                    pass
                else:
                    print(f"{param.numel():10} | {name}")
                    self.rough_total_trainable += param.numel()
            else:
                pass
        print_divider(l=50)
        print(f"{self.rough_total_trainable:10} | Total Trainable (Rough)")
        print_divider()

    def report_layerwise_wio(self):
        def analyzer_fn(module, input, output, modname):
            weight_shape = tuple(module.weight.shape) if hasattr(module, 'weight') else None
            if weight_shape is not None:
                input_shape = tuple(input[0].shape) 
                output_shape = tuple(output.shape)
                
                print(f"{modname:>50} | {str(weight_shape):>20} w| {str(input_shape):>20} i| {str(output_shape):>20} o|")

        hooklist=[]
        for n, m in self.model.named_modules():
            if len(list(m.children()))<=0:
                hooklist.append(m.register_forward_hook(partial(analyzer_fn, modname=n)))
        
        print_divider()
        print(f"[Info]: Reporting layerwise shape of weight, input, output...\n\n")
        device = next(self.model.parameters()).device
        with torch.no_grad():
            self.model(torch.rand(1, 2, 32, 32).to(device)) #TODO: generalize by parameterize model input tensor, hardcoded to csi dataset input
        print_divider()
        for hk in hooklist:
            hk.remove()


class Cost2100DataManager:
    def __init__(self, scenario, rawdir, batchsize) -> None:
        self.scenario = scenario
        keyword_dict = dict(indoor='in',outdoor='out')
        self.kw = keyword_dict[self.scenario]
        self.datadir = Path(rawdir)
        if not self.datadir.is_dir():
            raise ValueError(f"dataroot: {self.datadir} does not exist")

        self.bs = batchsize
        # NOTE: only load when it is needed to avoid unnecessary loading time
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None

        # TODO: assuming indoor case for now

    @property
    def train_loader(self):
        if self._train_loader is None:
            self.create_train_loader()
        return self._train_loader

    def create_train_loader(self):
        print(f"Creating - {self.scenario} - Train Loader ...")
        Htrainin_tensor = torch.FloatTensor(loadmat(self.datadir/f'DATA_Htrain{self.kw}.mat')['HT'])
        train_dataset = TensorDataset(Htrainin_tensor, Htrainin_tensor)
        self._train_loader = DataLoader(train_dataset, batch_size=self.bs, shuffle=True)
        self.niter_per_epoch = len(self._train_loader)

    @property
    def val_loader(self):
        if self._val_loader is None:
            self.create_val_loader()
        return self._val_loader

    def create_val_loader(self):
        print(f"Creating - {self.scenario} - Val Loader ...")
        Hvalin_tensor = torch.FloatTensor(loadmat(self.datadir/f'DATA_Hval{self.kw}.mat')['HT'])
        val_dataset = TensorDataset(Hvalin_tensor, Hvalin_tensor)
        self._val_loader = DataLoader(val_dataset, batch_size=self.bs, shuffle=False)
    
    @property
    def test_loader(self):
        if self._test_loader is None:
            self.create_test_loader()
        return self._test_loader

    def create_test_loader(self):
        print(f"Creating - {self.scenario} - Test Loader ...")
        Htestin_tensor = torch.FloatTensor(loadmat(self.datadir/f'DATA_Htest{self.kw}.mat')['HT'])
        test_dataset = TensorDataset(Htestin_tensor, Htestin_tensor)
        self._test_loader = DataLoader(test_dataset, batch_size=self.bs, shuffle=False)

class OutputManager:
    def __init__(self, outdir, label, wandb_proj='dl-csi'):
        # if not Path(outdir).exists():
        #     raise NotADirectoryError(f"{outdir} not found.")
        
        ts = datetime.now().strftime('%m%d_%H%M%S')
        
        self.run_id = f"{ts}_{label}" if label is not None else f"{ts}_run"
        self.rundir = Path(outdir).resolve() / self.run_id
        self.ckptdir = self.rundir / "ckpt"
        self.best_ckpt_pth = self.ckptdir / "best_ckpt.pth"
        self.tbdir = self.rundir / "tb"
        self._create_folders()
        self._start_tb()
        
        self.wandb_init = False
        if wandb_installed is True:
            wandb.init(project=wandb_proj, 
                       name=self.run_id,
                       dir=self.rundir)
            self.wandb_init = True

    def _create_folders(self):
        self.rundir.mkdir(parents=True, exist_ok=True)
        self.ckptdir.mkdir(parents=True, exist_ok=True)
        self.tbdir.mkdir(parents=True, exist_ok=True)       

    def _start_tb(self):
        self.tb_writer = SummaryWriter(self.tbdir)

    def save_args(self, args):
        print_divider()
        print(f"args:\n{json.dumps(vars(args), indent=True)}")
        with open(self.rundir/"args.json", "w") as outfile:
            json.dump(vars(args), outfile, indent=True)

    def save_model(self, model, epoch):
        torch.save(model.state_dict(), self.best_ckpt_pth)

    def log(self, d, step):
        for k, v in d.items():
            self.tb_writer.add_scalar(k, v, step)
        
        if self.wandb_init is True:
            wandb.log(d, step=step)
    
    def backup_scripts(self, script_dir):
        #TODO: can be implemented in a better way
        #TODO: folder stuctures are not aligned but enough as backup for now
        # hardcode for now, just main.py, utils/, model/
        # assumption of usage model is that we only add new model definition to model/ folder
        # and some changes to main.py
        # utils are more or less stable
        to_backup = ['main.py', 'model/', 'utils/']

        script_backup_dir = self.rundir/"run_scripts"
        script_backup_dir.mkdir(parents=True, exist_ok=True)

        for doc in to_backup:
            if '/' in doc:
                shutil.copytree('/'.join([script_dir, doc]), script_backup_dir, dirs_exist_ok=True)
            else:
                shutil.copy2('/'.join([script_dir, doc]), script_backup_dir)


def calculate_nmse_and_rho(H, H_hat):
    """
    Calculate NMSE and rho between original and reconstructed CSI, correctly handling FFT and IFFT operations.

    Parameters:
    - H: Original CSI tensor, spatial domain (batch_size, 2, height, width).
    - H_hat: Reconstructed CSI tensor, spatial domain (batch_size, 2, height, width).
    """
    # Assuming H and H_hat are real-valued and on the same device.
    device = H.device

    # Normalize H and H_hat if not already normalized
    H_normalized = H - 0.5
    H_hat_normalized = H_hat - 0.5

    # Combine real and imaginary parts to form complex tensors
    H_complex = torch.view_as_complex(H_normalized.permute(0, 2, 3, 1).contiguous())
    H_hat_complex = torch.view_as_complex(H_hat_normalized.permute(0, 2, 3, 1).contiguous())

    # Perform FFT on both tensors
    H_fft = torch.fft.fftn(H_complex, dim=[1, 2])
    H_hat_fft = torch.fft.fftn(H_hat_complex, dim=[1, 2])

    # Calculate NMSE
    nmse_num = torch.sum(torch.abs(H_fft - H_hat_fft) ** 2, dim=[1, 2])
    nmse_den = torch.sum(torch.abs(H_fft) ** 2, dim=[1, 2])
    nmse = torch.mean(nmse_num / nmse_den)
    nmse_dB = 10 * torch.log10(nmse)

    # Calculate rho (spectral correlation coefficient)
    rho_num = torch.sum(torch.conj(H_fft) * H_hat_fft, dim=[1, 2])
    rho_den = torch.sqrt(torch.sum(torch.abs(H_fft) ** 2, dim=[1, 2]) * torch.sum(torch.abs(H_hat_fft) ** 2, dim=[1, 2]))
    rho = torch.mean(torch.abs(rho_num) / rho_den)

    return nmse.item(), nmse_dB.item(), rho.item()
