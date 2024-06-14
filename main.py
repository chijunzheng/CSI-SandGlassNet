import os
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import argparse
from pathlib import Path
from model.csinet import CsiNet
from model.SandGlassNet import SandGlassNet
from utils.utils import ModelAnalyzer, Cost2100DataManager, OutputManager
from utils.utils import calculate_nmse_and_rho, print_divider
from utils.meter import AverageMeter


DEFAULT_DATADIR="/content/drive/MyDrive/2.Learning/Masters_Thesis/COST2100"
DEFAULT_RUNDIR="./runs"
DEFAULT_WANDB_PROJ="dl-csi"
DEFAULT_BS = 200
DEVICE_LIST=['cpu', 'cuda']
DATA_SCENARIOS=['indoor', 'outdoor']
CR_LIST=[4, 8, 16, 32]
DEFAULT_NEPOCH = 1000
DEFAULT_LR = 0.001


def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning for CSI")
    
    parser.add_argument('--batchsize', type=int, default=DEFAULT_BS,
                        help=f'Batch Size. Default is {DEFAULT_BS}.')
    
    parser.add_argument('--nepoch', type=int, default=DEFAULT_NEPOCH,
                        help=f'number training epoch. Default is {DEFAULT_NEPOCH}')
    
    parser.add_argument('--lr', type=float, default=DEFAULT_LR,
                        help=f'learning rate. Default is {DEFAULT_LR}')
    
    parser.add_argument('--test_only', action='store_true', help='evaluate checkpoint')
    
    parser.add_argument('--ckpt', type=str, default=None,
                        help='checkpoint to be evaluated, only effective with --test_only')
    
    parser.add_argument('--device', choices=DEVICE_LIST, default=DEVICE_LIST[-1], 
                        help=f'any of {DEVICE_LIST}')
    
    parser.add_argument('--datadir', type=str, default=DEFAULT_DATADIR,
                        help='path to the root of COST2100 raw data')

    parser.add_argument('--data_scenario', type=str, default=DATA_SCENARIOS[0],
                        help=f'data environment of COST2100 raw data, any of {DATA_SCENARIOS}')

    parser.add_argument('--cr', type=int, choices=CR_LIST, default=CR_LIST[0], 
                        help=f'compression ratio of CSI, any of {CR_LIST}')
        
    parser.add_argument('--rundir', type=str, default=DEFAULT_RUNDIR,
                        help='path to the output of run')
    
    parser.add_argument('--wandb_proj', type=str, default=DEFAULT_WANDB_PROJ,
                        help='project in wandb')
    
    parser.add_argument('--label', type=str, default=None,
                        help='Optional label for the run. Defaults to None. Output directory is always prefixed by date time')
    
    return parser.parse_args()


def create_model(parsed_args):
    # model = CsiNet()
    data_h_or_w = 32
    compression_ratio = parsed_args.cr
    data_channel = 2
    stage1_emb = 64
    stage2_emb = 32
    patch_size = 2
    num_transformer_per_stage = 1
    num_attn_head = 4
    ffn_expansion_ratio = 4
    dropout_rate = 0.1

    model = SandGlassNet(
        data_h_or_w, data_channel, compression_ratio, 
        stage1_emb, stage2_emb, patch_size, 
        num_transformer_per_stage, num_attn_head, 
        ffn_expansion_ratio, dropout_rate)

    return model


def evaluate_loop(model, data: Cost2100DataManager, eval_type): #type is either valid or test
    if eval_type == 'val':
        loader = data.val_loader
    elif eval_type == 'test':
        loader = data.test_loader
    else:
        raise ValueError("eval_type only accept either val or test, revise caller")

    model.eval()
    device = next(model.parameters()).device

    criterion = nn.MSELoss()
    loss_meter = AverageMeter(f'{eval_type}_loss')

    recovered_csi_list, original_csi_list = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(-1, 2, 32, 32)
            targets = targets.view(-1, 2, 32, 32)

            _, outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_meter.update(loss.item())

            recovered_csi_list.append(outputs)
            original_csi_list.append(targets)

    original_csi_tensor = torch.concat(original_csi_list)
    recovered_csi_tensor = torch.concat(recovered_csi_list)

    # Calculate NMSE and Rho
    nmse, nmse_dB, rho = calculate_nmse_and_rho(original_csi_tensor, recovered_csi_tensor)

    return loss_meter, nmse, nmse_dB, rho


def main():
    args = parse_args()
    device = torch.device(args.device)

    # by design, let's ensure model creation before creating unnecessary collaterals
    model = create_model(args)
    model.to(device)

    data = Cost2100DataManager(args.data_scenario, args.datadir, args.batchsize)

    omgr = OutputManager(outdir=args.rundir, label=args.label, wandb_proj=args.wandb_proj)
    omgr.save_args(args)
    omgr.backup_scripts(os.path.dirname(os.path.abspath(__file__)))

    omgr.log({"Model/Environment":DATA_SCENARIOS.index(args.data_scenario)+1}, 0) # 1 is indoor, 2 is outdoor

    analyzer = ModelAnalyzer(model)
    analyzer.print_model_arch()
    analyzer.report_layerwise_parameters()
    omgr.log({"Model/Rough_Params":analyzer.rough_total_trainable}, 0)
    analyzer.report_layerwise_wio()
    
    if args.test_only is not True:
        # Training loop
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()

        best_valid_loss = float('inf') 

        for epoch in range(args.nepoch):
            model.train()
            
            train_loss_meter = AverageMeter(f"train_loss")
            for it, (inputs, targets) in enumerate(data.train_loader):
                global_iter = epoch*data.niter_per_epoch + it + 1
                frac_epoch = global_iter/data.niter_per_epoch

                inputs, targets = inputs.to(device), targets.to(device)

                inputs = inputs.reshape(-1, 2, 32, 32)
                targets = targets.reshape(-1, 2, 32, 32)

                optimizer.zero_grad()

                _, outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss_meter.update(loss.item())
                omgr.log({'Iter/Train/loss': loss.item()}, global_iter)
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                omgr.log({'Iter/Train/lr': lr}, global_iter)
                
            # End of Epoch
            val_loss_meter, val_nmse, val_nmse_dB, val_rho  = evaluate_loop(model, data, 'val')
            epoch_dict = {
                'Epoch/Epoch': frac_epoch,
                'Epoch/Train/loss': train_loss_meter.avg,
                'Epoch/Val/loss': val_loss_meter.avg,
                'Epoch/Val/nmse': val_nmse,
                'Epoch/Val/nmse_dB': val_nmse_dB,
                'Epoch/Val/rho': val_rho,
            }
            omgr.log(epoch_dict, global_iter)

            logstr = (f"[Train] {epoch:4}/{args.nepoch} epoch => "
                      f"{train_loss_meter.avg:12.8f} train_loss, "
                      f"{val_loss_meter.avg:12.8f} val_loss, "
                      f"{val_nmse:10.6f} val_nmse, "
                      f"{val_nmse_dB:10.3f} val_nmse_dB, "
                      f"{val_rho:10.2f} val_rho")
            
            if val_loss_meter.avg < best_valid_loss:
                best_valid_loss = val_loss_meter.avg
                omgr.save_model(model, epoch)
                logstr += f" ...... (new best ckpt! epoch id {epoch})"

            print(logstr)
        
        # End of all epochs (End of Training)
        print_divider()
        print(f"\n[Test] End of Training, loading best ckpt: {omgr.best_ckpt_pth}")
        model.load_state_dict(torch.load(omgr.best_ckpt_pth))

    else:
        print_divider()
        if args.ckpt is not None:
            # test only with ckpt provided
            model_path = Path(args.ckpt)
            print(f"\n[Test] Loading ckpt: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            # test only with no load
            print("[Test] Model will be tested with random initialized states")

    # Test
    model.to(device)
    test_loss_meter, test_nmse, test_nmse_dB, test_rho = evaluate_loop(model, data, 'test')

    epoch_dict = {
        'Test/loss': test_loss_meter.avg,
        'Test/nmse': test_nmse,
        'Test/nmse_dB': test_nmse_dB,
        'Test/rho': test_rho,
    }
    omgr.log(epoch_dict, global_iter)
    
    print(f'[Test] {test_loss_meter.name}: {test_loss_meter.avg:.9f}')
    print(f'[Test] Metrics/NMSE: {test_nmse:.5f}')
    print(f'[Test] Metrics/NMSE_dB: {test_nmse_dB:.2f}')
    print(f'[Test] Metrics/Rho: {test_rho:3f}')
    print("end of script.")

if __name__ == "__main__":
    main()
