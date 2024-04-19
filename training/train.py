import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import ClipLoss, get_cast_dtype
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    lambda_reg_l2 = args.lambdaRegL2

    model.train()
    loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod,
        lambda_reg_l2=lambda_reg_l2)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    ### Jiahao
    # with autograd.detect_anomaly():
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            
            # print("accum_freq 1 activated activated")
            with autocast():
                #print("#--------------------------Modified by me; line 97 activated--------------------------#")
                image_features, text_features, logit_scale, mu_img, sigma_img, mu_txt, sigma_txt = model(images, texts,device)
                # raise value error if image_features contains,text-features, mu_img, sigma_img, mu_txt, sigma_txt contains NaN
                # and print which one contains NaN
                # if torch.any(torch.isnan(image_features)):
                #     raise ValueError("image_features contains NaN in model.py line 102")
                # if torch.any(torch.isnan(text_features)):
                #     raise ValueError("text_features contains NaN in model.py line 104")
                # if torch.any(torch.isnan(mu_img)):
                #     raise ValueError("mu_img contains NaN in model.py line 106")
                # if torch.any(torch.isnan(sigma_img)):
                #     raise ValueError("sigma_img contains NaN in model.py line 108")
                # if torch.any(torch.isnan(mu_txt)):
                #     raise ValueError("mu_txt contains NaN in model.py line 110")
                # if torch.any(torch.isnan(sigma_txt)):
                #     raise ValueError("sigma_txt contains NaN in model.py line 112")
                
                total_loss,avg_norm,l2_reg = loss(image_features, text_features, logit_scale, mu_img, sigma_img, mu_txt, sigma_txt)
            if i == 0:
                print("avg_norm: ",avg_norm)
                print("l2_reg: ",l2_reg)
            backward(total_loss, scaler)

            # from pdb import set_trace
            # set_trace()

            # loop through the gradients of all the parameters and check nan
            # count = 0

            ### JIAHAO FEEDBACK: model.named_parameters do not return the parameters in order of the actual model ###
            ### https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html
            ### Register hooks to every single module and detect the NaN values in the order the model is executed ###
            

            ##### 
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         # check if name contains SIGMA_Images or SIGMA_texts
            #         if "SIGMA_Images" in name or "SIGMA_Texts" in name:
            #         # check if the name ends with .9 or .10
            #             if ".9" in name or ".10" in name:
            #                 # print out the name and min and max of the param
            #                 if torch.distributed.get_rank() == 0:
            #                     pass
            #                     # print("###### train 136 ######")
            #                     # print("####### 137 param.grad",name, "min:", torch.min(param.grad).item(), "max:", torch.max(param.grad).item())
            #                     # print("####### 138 param.data",name, "min:", torch.min(param.data).item(), "max:", torch.max(param.data).item())

    
            #             if torch.isnan(param.grad).any():
            #                 print(name, "contains NaN values!")
            #                 raise ValueError(f"NaN value detected in gradients Stopping training.")



            # my_checkpoint_dict = {
            #     "epoch": epoch,
            #     "name": args.name,
            #     "state_dict": model.state_dict(),
            #     "optimizer": optimizer.state_dict(),
            # }

            # parent_dir = '/share/data/pals/kevin/open_clip/src/checkpoints_Jiahao'

            # if i > 5:
            #     previous_path = os.path.join(parent_dir, f"checkpoint_{epoch}_{i-5}.pt")
            #     if os.path.exists(previous_path):
            #         time.sleep(3)
            #         os.remove(previous_path)
            #         print("removed previous checkpoint at", previous_path)
            #     else: 
            #         print("previous checkpoint does not exist")

            ######## Continuous Saving of each Step ########
            # path = os.path.join(parent_dir, f"checkpoint_{epoch}_jiahao.pt")
            # if os.path.exists(parent_dir):
            #     torch.save(my_checkpoint_dict, path)
            #     print("saved at", path)
            # else:
            #     print("path does not exist")
            # print("############finishing step:", i, "############")
            ################################################
        else:
            ##################################################################
            # Commented Out by me
            print("#--------------------------LINE 103 ACTIVATED: Else engaged--------------------------#")
            # # First, cache the features without any gradient tracking.
            # with torch.no_grad():
            #     with autocast():
            #         chunk_image_features, chunk_text_features, _ = model(images, texts)
            #     accum_image_features.append(chunk_image_features)
            #     accum_text_features.append(chunk_text_features)

            #     accum_images.append(images)
            #     accum_texts.append(texts)

            # # If (i + 1) % accum_freq is not zero, move on to the next batch.
            # if ((i + 1) % args.accum_freq) > 0:
            #     # FIXME this makes data time logging unreliable when accumulating
            #     continue

            # # Now, ready to take gradients for the last accum_freq batches.
            # # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # # Call backwards each time, but only step optimizer at the end.
            # optimizer.zero_grad()
            # for j in range(args.accum_freq):
            #     images = accum_images[j]
            #     texts = accum_texts[j]
            #     with autocast():
            #         chunk_image_features, chunk_text_features, logit_scale = model(images, texts)
            #         image_features = torch.cat(
            #             accum_image_features[:j] + [chunk_image_features] + accum_image_features[j + 1:])
            #         text_features = torch.cat(
            #             accum_text_features[:j] + [chunk_text_features] + accum_text_features[j + 1:])
            #         total_loss = loss(image_features, text_features, logit_scale)
            #     backward(total_loss, scaler)
            ##################################################################

        ##### Edit 10/1 for Jiahao--going through parameters to see which one produces nan #####

        # raise ValueError(f"NaN value detected in gradients Stopping training.")
    
        #######################################################################################


        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        ################Added by me 09/13 Jiahao Advcie for checking NaN weights##################
        # for name, param in model.state_dict().items():
        #     if torch.isnan(param).any():
        #         print(name, "contains NaN values!")
        
        ##################################



        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        ##################################################################
        # Commented Out by me
        # with torch.no_grad():
        #     unwrap_model(model).logit_scale.clamp_(0, math.log(100))
        ##################################################################

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            ##################################################################
            # Commented Out by me
            # logit_scale_scalar = logit_scale.item()
            logit_scale_scalar = 0.0
            ##################################################################
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.accum_freq * args.batch_size * args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": args.accum_freq * args.batch_size * args.world_size / batch_time_m.val,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    if args.zero_shot_evaluate:
        zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
        quit()
    
    if (epoch % args.zeroshot_frequency) == 0:
        zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
        metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):

        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    ##################################################################
                    # Commented Out by me
                    # image_features, text_features, logit_scale = model(images, texts)
                    # # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # # however, system RAM is easily exceeded and compute time becomes problematic
                    # all_image_features.append(image_features.cpu())
                    # all_text_features.append(text_features.cpu())
                    # logit_scale = logit_scale.mean()
                    # logits_per_image = logit_scale * image_features @ text_features.t()
                    # logits_per_text = logits_per_image.t()

                    # batch_size = images.shape[0]
                    # labels = torch.arange(batch_size, device=device).long()
                    # total_loss = (
                    #     F.cross_entropy(logits_per_image, labels) +
                    #     F.cross_entropy(logits_per_text, labels)
                    # ) / 2
                    ##################################################################
                    batch_size = images.shape[0]
                    image_features, text_features, logit_scale, mu_img, sigma_img, mu_txt, sigma_txt = model(images, texts,device)
                    def calculate_P_lessMemory(Input,Mu,Sigma,device,b,d):
                        pi = 3.14159265358979323846
                        Q = torch.empty(0, b).to(device)
                        for i in range(Mu.shape[0]):
                            Q = torch.cat((Q, torch.sum(torch.square(Input - Mu[i])*Sigma[i], dim=1).reshape(1,b)), dim=0)
                        return (Q - torch.sum(torch.log(Sigma), dim=1).reshape(-1,1) + d*torch.log(torch.tensor(2*pi).to(device)))/2

                    def calculate_loss_less(P,device):
                        # find the main diagonal of res1
                        diag = torch.diag(P).to(device)

                        return torch.sum(torch.max(torch.max(P, dim=0).values - diag, torch.tensor(0))) \
                            + torch.sum(torch.max(torch.max(P, dim=1).values - diag, torch.tensor(0)))                    
                    
                    b = image_features.shape[0]
                    d = image_features.shape[1]

                    P_img = calculate_P_lessMemory(image_features, mu_img, sigma_img, device,b,d)
                    P_txt = calculate_P_lessMemory(text_features, mu_txt, sigma_txt, device,b,d)

                    # if P_img contains NaN print "P_img contains NaN"
                    if torch.isnan(P_img).any():
                        print("P_img contains NaN")
                    if torch.isnan(P_txt).any():
                        print("P_txt contains NaN")

                    loss_img = calculate_loss_less(P_img,device)
                    loss_txt = calculate_loss_less(P_txt,device)

                    if torch.isnan(loss_img).any():
                        print("loss_img contains NaN")
                    if torch.isnan(loss_txt).any():
                        print("loss_txt contains NaN")
                        
                    total_loss = (loss_img + loss_txt)
                    ##################################################################

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size

                # print(f"Loss: {cumulative_loss / num_samples:.6f}\t")

                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
