import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm

from open_clip import get_cast_dtype, get_tokenizer
from .precision import get_autocast
from .imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template


def zero_shot_classifier(model, classnames, templates, args):
    tokenizer = get_tokenizer(args.model)
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenizer(texts).to(args.device)  # tokenize
            if args.distributed and not args.horovod:
                class_embeddings = model.module.encode_text(texts)
            else:
                class_embeddings = model.encode_text(texts)

            # find log probs
            
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

pi = 3.14159265358

def calculate_P_lessMemory(Input,Mu,Sigma,device,b,d):
    Q = torch.empty(0, b).to(device)
    for i in range(Mu.shape[0]):
        Q = torch.cat((Q, torch.sum(torch.square(Input - Mu[i])*Sigma[i], dim=1).reshape(1,b)), dim=0)
    return (Q - torch.sum(torch.log(Sigma), dim=1).reshape(-1,1) + d*torch.log(torch.tensor(2*pi).to(device)))/2

def run(model, text_embeds, dataloader, args):
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(args.device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                if args.distributed and not args.horovod:
                    img_embeds = model.module.encode_image(images)
                else:
                    img_embeds = model.encode_image(images)
                img_embeds = F.normalize(img_embeds, dim=-1)
                # find log probs
                mu_img, sigma_img, mu_txt, sigma_txt = model.module.mu_sigma_img_txt(img_embeds, text_embeds)

                b = img_embeds.shape[0]
                d = img_embeds.shape[1]
                device = args.device

                P_img = calculate_P_lessMemory(img_embeds, mu_img, sigma_img, device,b,d)
                P_txt = calculate_P_lessMemory(text_embeds, mu_txt, sigma_txt, device,b,d)

                log_probs = P_img + P_txt.T

            ###
            # measure accuracy
            acc1, acc5 = accuracy(log_probs, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def zero_shot_eval(model, data, epoch, args):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}

    logging.info('Starting zero-shot imagenet.')

    logging.info('Building zero-shot classifier')
    text_embeds = zero_shot_classifier(model, imagenet_classnames, openai_imagenet_template, args)

    logging.info('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        top1, top5 = run(model, text_embeds, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, text_embeds, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5

    logging.info('Finished zero-shot imagenet.')

    return results
