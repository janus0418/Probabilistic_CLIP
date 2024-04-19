import logging
import pickle
import torch
import torch.nn.functional as F
import time
import pandas as pd

from tqdm import tqdm
from PIL import Image
from open_clip import get_cast_dtype, get_tokenizer
from .precision import get_autocast
from .imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
import torch.utils.data as dutils
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CocoCaptions

PROBABILISTIC_CLIP = False
EVALUATE = False
SEEING_WHICH_COCO_EXAMPELS_FAIL = False
class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename,sep=sep)
            
        # print(df.head())

        # print(input_filename)

        self.images = df['filepath'].tolist()
        self.captions = df['caption'].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


def get_csv_dataset(args,model,preprocess_fn):
    input_file_path = "greg_comp.csv"
    tokenizer = get_tokenizer(args.model)
    dataset = CsvDataset(
        input_file_path,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    return dataset


def greg_comp_data(args, model, preprocess_fn):
    input_file_path = "/share/data/pals/kevin/open_clip/src/kevin_3x3_comparisons/complex_coco_preprocessed.pkl"
    tokenizer = get_tokenizer(args.model)

    # open the pickle file
    with open(input_file_path, 'rb') as f:
        df = pickle.load(f)

    image_filepaths = df['filepath'][0:9].tolist()
    captions = df['caption'].tolist()

    images = []
    for i in range(len(image_filepaths)):
        images.append(preprocess_fn(Image.open(image_filepaths[i])))
    
    texts = []
    for i in range(len(captions)):
        texts.append(tokenizer([captions[i]])[0])

    # print shape of image and text tensor representations
    print("length of image_encodings: ", len(images))
    print("length of text_encodings: ", len(texts))

    device = args.device
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    image_encodings = []
    text_encodings = []

    images_dataloader = DataLoader(images, batch_size=9, shuffle=False)
    texts_dataloader = DataLoader(texts, batch_size=13, shuffle=False)
    for images in images_dataloader:
        for texts in texts_dataloader:
            images = images.to(args.device) # for batch of 100, 20 images
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            
            texts = texts.to(args.device) # for batch of 100, 100 texts 
            if cast_dtype is not None:
                texts = texts.to(dtype=cast_dtype)
            
            with autocast():
            ######USE FOR PROBABILISTIC CLIP####################################
            #     if args.distributed and not args.horovod:
            #         img_embeds = model.module.encode_image(images)
            #     else:
            #         img_embeds = model.encode_image(images)
            #     img_embeds = F.normalize(img_embeds, dim=-1) # 20 x 1024
            #     image_encodings.append(img_embeds)

            #     if args.distributed and not args.horovod:
            #         text_embeds = model.module.encode_text(texts)
            #     else:
            #         text_embeds = model.encode_text(texts)
            #     text_embeds = F.normalize(text_embeds, dim=-1)
            #     text_encodings.append(text_embeds)

            #     mu_img_batch, sigma_img_batch = model.module.mu_sigma_img(text_embeds, device) # 100 x 1024
            #     mu_txt_batch, sigma_txt_batch = model.module.mu_sigma_txt(img_embeds, device) # 20 x 1024

            #     mus_img.append(mu_img_batch)
            #     sigmas_imgs.append(sigma_img_batch)
            #     mus_txts.append(mu_txt_batch)
            #     sigmas_txts.append(sigma_txt_batch)

            # image_encodings = torch.cat(image_encodings)
            # text_encodings = torch.cat(text_encodings)
            # mu_img = torch.cat(mus_img)
            # sigma_img = torch.cat(sigmas_imgs)
            # mu_txt = torch.cat(mus_txts)
            # sigma_txt = torch.cat(sigmas_txts)

            # P_img = calculate_P_lessMemory(image_encodings, mu_img, sigma_img)
            # P_txt = calculate_P_lessMemory(text_encodings, mu_txt, sigma_txt)

            # # print shape of P_img and P_txt
            # print("shape of P_img: ", P_img.shape)
            # print("shape of P_txt: ", P_txt.shape)

            # # text to image (same text different images)
            # P_img = torch.softmax(-P_img, dim=1)
            # P_txt = torch.softmax(-P_txt, dim=0)

            # P_combinded_text_2_image = torch.softmax(P_img+P_txt.T, dim=1)

            # # convert P_img and P_txt to cpu
            # P_img = P_img.cpu()
            # P_txt = P_txt.cpu()
            # P_combinded_text_2_image = P_combinded_text_2_image.cpu()
            # # save P_img and P_txt to csv
            # P_img_df = pd.DataFrame(P_img.detach().numpy())
            # P_txt_df = pd.DataFrame(P_txt.detach().numpy())
            # P_combinded_text_2_image_df = pd.DataFrame(P_combinded_text_2_image.detach().numpy())
            # save_path = "/share/data/pals/kevin/open_clip/src/kevin_3x3_comparisons"
            # P_img_df.to_csv(save_path + "/P_img.csv", index=False)
            # P_txt_df.to_csv(save_path + "/P_txt.csv", index=False)
            # P_combinded_text_2_image_df.to_csv(save_path + "/P_combinded_text_2_image.csv", index=False)
            ####################################################################
                if args.distributed and not args.horovod:
                    img_embeds = model.module.encode_image(images)
                else:
                    img_embeds = model.encode_image(images)
                img_embeds = F.normalize(img_embeds, dim=-1) # 20 x 1024
                image_encodings.append(img_embeds)

                if args.distributed and not args.horovod:
                    text_embeds = model.module.encode_text(texts)
                else:
                    text_embeds = model.encode_text(texts)
                text_embeds = F.normalize(text_embeds, dim=-1)
                text_encodings.append(text_embeds)
            image_encodings = torch.cat(image_encodings)
            text_encodings = torch.cat(text_encodings)

            P_img = image_encodings @ text_encodings.T
            P_txt = text_encodings @ image_encodings.T

            P_img = P_img * 100
            P_txt = P_txt * 100

            print("shape of P_img: ", P_img.shape)
            print("shape of P_txt: ", P_txt.shape)

            P_img = torch.softmax(P_img, dim=1)
            P_txt = torch.softmax(P_txt, dim=0)

            P_combinded_text_2_image = torch.softmax(P_img+P_txt.T, dim=1)

            # convert P_img and P_txt to cpu
            P_img = P_img.cpu()
            P_txt = P_txt.cpu()
            P_combinded_text_2_image = P_combinded_text_2_image.cpu()
            # save P_img and P_txt to csv
            P_img_df = pd.DataFrame(P_img.detach().numpy())
            P_txt_df = pd.DataFrame(P_txt.detach().numpy())
            P_combinded_text_2_image_df = pd.DataFrame(P_combinded_text_2_image.detach().numpy())
            save_path = "/share/data/pals/kevin/open_clip/src/kevin_3x3_comparisons"
            P_img_df.to_csv(save_path + "/P_img.csv", index=False)
            P_txt_df.to_csv(save_path + "/P_txt.csv", index=False)
            P_combinded_text_2_image_df.to_csv(save_path + "/P_combinded_text_2_image.csv", index=False)

            print("greg_comp_data done")
            quit()
    return P_img


def greg_comp_probabilistic_clip(args, model, preprocess_fn):
    autocast = get_autocast(args.precision)
    
    dataset = get_csv_dataset(args,model,preprocess_fn)
    (image_encodings, text_encodings,
                mu_img, sigma_img,
                mu_txt, sigma_txt, 
                text_to_image_map, image_to_text_map) = encode_dataset(args, model, dataset)
    with autocast():
        # text-to-image recall
        P_img = calculate_P_lessMemory(image_encodings, mu_img, sigma_img)
        # P_txt = calculate_P_lessMemory(text_encodings, mu_txt, sigma_txt)

        P_img_df = pd.DataFrame(P_img.numpy())
        save_path = "/share/data/pals/kevin/open_clip/src/kevin_3x3_comparisons"
        P_img_df.to_csv(save_path + "/P_img.csv", index=False)
    return P_img

"""
image_encodings, text_encodings, text_to_image_map, image_to_text_map
"""

def greg_comp_original_clip(args, model, preprocess_fn):
    autocast = get_autocast(args.precision)
    dataset = get_csv_dataset(args,model,preprocess_fn)
    image_encodings, text_encodings, text_to_image_map, image_to_text_map = encode_dataset_for_original_clip(args, model, dataset)
    print("Text-to-image recall...")
    with autocast():
        P_img = text_encodings @ image_encodings.T
        P_img_df = pd.DataFrame(P_img.numpy())
        save_path = "/share/data/pals/kevin/open_clip/src/kevin_3x3_comparisons"
        P_img_df.to_csv(save_path + "/P_img.csv", index=False)
    return P_img        


def zero_shot_classifier(model, classnames, templates, args):
    tokenizer = get_tokenizer(args.model)
    with torch.no_grad():
        zeroshot_weights = []
        # print("shape of classnames: ", len(classnames))
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
    if PROBABILISTIC_CLIP == True:
        pred = output.topk(max(topk), 1, False, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
    else:
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]



pi = 3.14159265358

def calculate_P_lessMemory(input,mu,sigma):
    """
    # Input = batch_input x 1024(dimension)
    # mu = batch_given x 1024(dimension)
    # Sigma = batch_given x 1024(dimension)  

    outputs = batch_input x batch_given

    hence, it is of the form:
    (input_1|given_1) ... (input_1|given_n)
    (input_2|given_1) ... (input_2|given_n)
    ...
    (input_n|given_1) ... (input_n|given_n)
    """
    input_squared = torch.pow(input,2) @ (1/(sigma + 1e-6)).T # [batch_input x batch_sig]

    mu_squared = torch.sum(torch.pow(mu,2) / (sigma + 1e-6), dim=1).reshape(1,mu.shape[0]) # 1 x batch_mu
    
    middle = input @ torch.div(mu,sigma + 1e-6).T # [batch_input x batch_sig]

    return (-2)*middle + input_squared + mu_squared + torch.sum(torch.log(sigma+ 1e-6), dim=1)

def encode_dataset_for_original_clip(args, model, dataset: dutils.Dataset, batch_size = 16):
    device = args.device
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    with torch.no_grad():
        # image_to_text_map[i] gives the corresponding text indices for the ith image
        #  (as there are multiple pieces of text for each image)
        image_to_text_map = []

        # text_to_image_map[i] gives the corresponding image index for the ith text
        text_to_image_map = []

        dataloader = dutils.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        image_encodings = []
        text_encodings = []

        text_index = 0
        image_index = 0

        for images, text in dataloader:
            images = images.to(device)
            text = text.to(device)

            # text has shape B x 5 x 77
            batch_size, captions_per_image, _ = text.shape

            # Update text_to_image_map and image_to_text_map for this batch
            for i in range(batch_size):
                # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                text_indices = list(range(text_index, text_index + captions_per_image))
                image_to_text_map.append(text_indices)
                text_index += captions_per_image

                # Each of the next captions_per_image text captions correspond to the same image
                text_to_image_map += [image_index] * captions_per_image
                image_index += 1

            # B x 5 x 77 -> (B*5) x 77
            text = torch.flatten(text, start_dim=0, end_dim=1)
            
            image_encodings.append(model.encode_image(images))
            text_encodings.append(model.encode_text(text))

        image_encodings = torch.cat(image_encodings)
        text_encodings = torch.cat(text_encodings)
        text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
        image_to_text_map = torch.LongTensor(image_to_text_map).to(device)

        # Normalise encodings
        image_encodings = image_encodings / image_encodings.norm(dim=-1, keepdim=True)
        text_encodings = text_encodings / text_encodings.norm(dim=-1, keepdim=True)

        return image_encodings, text_encodings, text_to_image_map, image_to_text_map

if PROBABILISTIC_CLIP == True:
    ####### PROBABILISTIC CLIP ############################
    def encode_dataset(args, model, dataset: dutils.Dataset, batch_size = 16):
        device = args.device
        autocast = get_autocast(args.precision)
        cast_dtype = get_cast_dtype(args.precision)
        with torch.no_grad():
            # image_to_text_map[i] gives the corresponding text indices for the ith image
            #  (as there are multiple pieces of text for each image)
            image_to_text_map = []

            # text_to_image_map[i] gives the corresponding image index for the ith text
            text_to_image_map = []

            dataloader = dutils.DataLoader(dataset, batch_size=batch_size, shuffle=False)

            image_encodings = []
            text_encodings = []

            mus_img = []
            sigmas_imgs = []
            mus_txts = []
            sigmas_txts = []

            text_index = 0
            image_index = 0

            for images, texts in dataloader:

                images = images.to(args.device) # for batch of 100, 20 images
                if cast_dtype is not None:
                    images = images.to(dtype=cast_dtype)
                
                texts = texts.to(args.device) # for batch of 100, 100 texts 
                if cast_dtype is not None:
                    texts = texts.to(dtype=cast_dtype)

                # text has shape B x 5 x 77
                batch_size, captions_per_image, _ = texts.shape

                # Update text_to_image_map and image_to_text_map for this batch
                for i in range(batch_size):
                    # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                    text_indices = list(range(text_index, text_index + captions_per_image))
                    image_to_text_map.append(text_indices)
                    text_index += captions_per_image

                    # Each of the next captions_per_image text captions correspond to the same image
                    text_to_image_map += [image_index] * captions_per_image
                    image_index += 1

                # B x 5 x 77 -> (B*5) x 77
                texts = torch.flatten(texts, start_dim=0, end_dim=1)
                
                with autocast():
                    if args.distributed and not args.horovod:
                        img_embeds = model.module.encode_image(images)
                    else:
                        img_embeds = model.encode_image(images)
                    img_embeds = F.normalize(img_embeds, dim=-1) # 20 x 1024
                    image_encodings.append(img_embeds)

                    if args.distributed and not args.horovod:
                        text_embeds = model.module.encode_text(texts)
                    else:
                        text_embeds = model.encode_text(texts)
                    text_embeds = F.normalize(text_embeds, dim=-1)
                    text_encodings.append(text_embeds)

                    mu_img_batch, sigma_img_batch = model.module.mu_sigma_img(text_embeds, device) # 100 x 1024
                    mu_txt_batch, sigma_txt_batch = model.module.mu_sigma_txt(img_embeds, device) # 20 x 1024

                    mus_img.append(mu_img_batch)
                    sigmas_imgs.append(sigma_img_batch)
                    mus_txts.append(mu_txt_batch)
                    sigmas_txts.append(sigma_txt_batch)

            image_encodings = torch.cat(image_encodings)
            text_encodings = torch.cat(text_encodings)
            mu_img = torch.cat(mus_img)
            sigma_img = torch.cat(sigmas_imgs)
            mu_txt = torch.cat(mus_txts)
            sigma_txt = torch.cat(sigmas_txts)

            text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
            image_to_text_map = torch.LongTensor(image_to_text_map).to(device)


            return (image_encodings, text_encodings,
                    mu_img, sigma_img,
                    mu_txt, sigma_txt, 
                    text_to_image_map, image_to_text_map)

    def run_zero_shot_COCO(model, preprocess_val, args):
        autocast = get_autocast(args.precision)
        cast_dtype = get_cast_dtype(args.precision)
        tokenizer = get_tokenizer(args.model)
        device = args.device
        coco_root = "/share/data/vdata/coco/images/val2017"
        coco_ann_file = "/share/data/vdata/coco/annotations/captions_val2017.json"

        dataset = CocoCaptions(
            root=coco_root,
            annFile=coco_ann_file,
            transform=preprocess_val,
            # Note: almost all images have 5 captions, but 12/5000 have 6, and 1/5000 has 7 - I ignore these few extra captions.
            target_transform=lambda texts: tokenizer(texts[:5])
        )

        (image_encodings, text_encodings,
                    mu_img, sigma_img,
                    mu_txt, sigma_txt, 
                    text_to_image_map, image_to_text_map) = encode_dataset(args, model, dataset)
        with autocast():
            num_text = text_encodings.shape[0]
            num_im = image_encodings.shape[0]
            captions_per_image = image_to_text_map.shape[1]
            text_to_image_recall_over_inputs = []
            text_to_image_recall_over_givens = []
            image_to_text_recall_over_inputs = []
            image_to_text_recall_over_givens = []
            # text-to-image recall
            print("Text-to-image recall...")

            P_img = calculate_P_lessMemory(image_encodings, mu_img, sigma_img) # 100 x 20
            
            # dist_matrix = text_encodings @ image_encodings.T  # dist_matrix[i] gives logits for ith text
            
            # dist_matrix = P_img # 5000 x 25000
            dist_matrix = P_img.T # 25000 x 5000

            # Note: this matrix is pretty big (5000 x 25000 with dtype float16 = 250MB)
            #  torch.argsort runs out of memory for me (6GB VRAM) so I move to CPU for sorting
            dist_matrix = dist_matrix.cpu()

            # Sort in reverse descending order; first is the smallest logit(negative log prob)
            inds = torch.argsort(dist_matrix, dim=1, descending=False)
            inds = inds.to(device)

            # shape and first few values of text_to_image_map and image_to_text_map
            # print("shape of text_to_image_map: ", text_to_image_map.shape)
            # print("shape of image_to_text_map: ", image_to_text_map.shape)
            # print("text_to_image_map: ", text_to_image_map[10])
            # print("image_to_text_map: ", image_to_text_map[10])
            # print("shape of P_img: ", P_img.shape)
            # print("shape of image_encodings: ", image_encodings.shape)
            # print("shape of text_encodings: ", text_encodings.shape)

            k_vals=[1, 5]
            for k in k_vals:
                # Extract top k indices only
                topk = inds[:, :k]

                ## For checking over inputs --> needs P_img.T
                # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
                correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1) # image_to_text_map[:, i].unsqueeze(-1)

                num_correct = correct.sum().item()
                # image_to_text_recall_over_givens.append(num_correct / num_text)
                text_to_image_recall_over_inputs.append(num_correct / num_text)

            dist_matrix = dist_matrix.T # 5000 x 25000
            inds = torch.argsort(dist_matrix, dim=1, descending=False)
            inds = inds.to(device)
            for k in k_vals:
                topk = inds[:, :k]
                ### For checking over givens --> needs P_img
                # For each image, check whether one of the 5 relevant captions was retrieved
                # Check if image matches its ith caption (for i=0..4)
                correct = torch.zeros((num_im,), dtype=torch.bool).cuda()
                for i in range(captions_per_image): # captions_per_image = 5
                    contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
                    correct = torch.logical_or(correct, contains_index) 

                num_correct = correct.sum().item()
                #text_to_image_recall_over_givens.append(num_correct / num_im)
                image_to_text_recall_over_givens.append(num_correct / num_im)
            """
            ------------------------------------------------------------------------
            """
            # image-to-text recall
            print("Image-to-text recall...")

            P_txt = calculate_P_lessMemory(text_encodings, mu_txt, sigma_txt)
            # dist_matrix = P_txt # 25000 x 5000
            dist_matrix = P_txt.T # 5000 x 25000
            dist_matrix = dist_matrix.cpu()
            # Sort in descending order; first is the biggest logit
            inds = torch.argsort(dist_matrix, dim=1, descending=False)
            inds = inds.to(device)

            if SEEING_WHICH_COCO_EXAMPELS_FAIL == True:
                # target rows 
                targets = [2048, 4104, 12, 2060, 4108, 4111, 4116, 4118, 4120, 25]

                # extract the rows of P_txt corresponding to the target rows
                P_txt_targets = dist_matrix[targets, :]
                inds_targets = inds[targets, :]
                # save P_txt_targets to pickle file
                save_path = "/share/data/pals/kevin/open_clip/src/kevin_3x3_comparisons"
                with open(save_path + "/P_txt_targets.pkl", "wb") as f:
                    pickle.dump(P_txt_targets, f)
                with open(save_path + "/inds_targets.pkl", "wb") as f:  
                    pickle.dump(inds_targets, f)


            for k in k_vals:
                # Extract top k indices only
                topk = inds[:, :k]
                
                ### For checking over inputs --> needs P_txt.T
                # For each image, check whether one of the 5 relevant captions was retrieved
                # Check if image matches its ith caption (for i=0..4)
                correct = torch.zeros((num_im,), dtype=torch.bool).cuda()
                for i in range(captions_per_image):
                    contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
                    correct = torch.logical_or(correct, contains_index) 
                
                if EVALUATE == True:
                    print("shape of correct for image_to_text_recall_over_inputs: ", correct.shape)
                    # # find the index of the correct image, ie. where the index is non-zero in correct
                    correct_index = torch.nonzero(correct).squeeze(1)
                    print("shape of correct_index for image_to_text_recall_over_inputs: ", correct_index.shape)

                    # # save the correct_index as a pickle file
                    # save_path = "/share/data/pals/kevin/open_clip/src/kevin_3x3_comparisons"
                    # with open(save_path + f"/correct_index_p_txt_recall_over_inputs_top_{k}.pkl", "wb") as f:
                    #     pickle.dump(correct_index, f)

                num_correct = correct.sum().item()
                image_to_text_recall_over_inputs.append(num_correct / num_im)

            #

            dist_matrix = dist_matrix.T # 25000 x 5000
            # Sort in descending order; first is the biggest logit
            inds = torch.argsort(dist_matrix, dim=1, descending=False)
            inds = inds.to(device)
            for k in k_vals:
                topk = inds[:, :k]
                ### For checking over givens --> needs P_txt
                # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
                correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1) # image_to_text_map[:, i].unsqueeze(-1)

                if EVALUATE == True:
                    print("shape of correct for image_to_text_recall_over_givens: ", correct.shape)
                    correct_index = torch.nonzero(correct).squeeze(1)

                    print("shape of correct_index for image_to_text_recall_over_givens: ", correct_index.shape)

                    # # save the correct_index as a pickle file
                    # save_path = "/share/data/pals/kevin/open_clip/src/kevin_3x3_comparisons"
                    # with open(save_path + f"/correct_index_p_txt_recall_over_givens_top_{k}.pkl", "wb") as f:
                    #     pickle.dump(correct_index, f)

                num_correct = correct.sum().item()
                text_to_image_recall_over_givens.append(num_correct / num_text)

            print("Text-to-image Recall@K(over inputs)")
            for k, x in zip(k_vals, text_to_image_recall_over_inputs):
                print(f" R@{k}: {100*x:.2f}%")
            
            print("Text-to-image Recall@K(over givens)")
            for k, x in zip(k_vals, text_to_image_recall_over_givens):
                print(f" R@{k}: {100*x:.2f}%")

            print("Image-to-text Recall@K(over inputs)")
            for k, x in zip(k_vals, image_to_text_recall_over_inputs):
                print(f" R@{k}: {100*x:.2f}%")
            
            print("Image-to-text Recall@K(over givens)")
            for k, x in zip(k_vals, image_to_text_recall_over_givens):
                print(f" R@{k}: {100*x:.2f}%")

            print("Done.")
        return text_to_image_recall_over_inputs,text_to_image_recall_over_givens, image_to_text_recall_over_inputs, image_to_text_recall_over_givens    
    ####### PROBABILISTIC CLIP ############################
else:
    def encode_dataset(args, model, dataset: dutils.Dataset, batch_size = 16):
        device = args.device
        autocast = get_autocast(args.precision)
        cast_dtype = get_cast_dtype(args.precision)
        with torch.no_grad():
            # image_to_text_map[i] gives the corresponding text indices for the ith image
            #  (as there are multiple pieces of text for each image)
            image_to_text_map = []

            # text_to_image_map[i] gives the corresponding image index for the ith text
            text_to_image_map = []

            dataloader = dutils.DataLoader(dataset, batch_size=batch_size, shuffle=False)

            image_encodings = []
            text_encodings = []

            text_index = 0
            image_index = 0

            for images, texts in dataloader:

                images = images.to(args.device) # for batch of 100, 20 images
                if cast_dtype is not None:
                    images = images.to(dtype=cast_dtype)
                
                texts = texts.to(args.device) # for batch of 100, 100 texts 
                if cast_dtype is not None:
                    texts = texts.to(dtype=cast_dtype)

                # text has shape B x 5 x 77
                batch_size, captions_per_image, _ = texts.shape

                # Update text_to_image_map and image_to_text_map for this batch
                for i in range(batch_size):
                    # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                    text_indices = list(range(text_index, text_index + captions_per_image))
                    image_to_text_map.append(text_indices)
                    text_index += captions_per_image

                    # Each of the next captions_per_image text captions correspond to the same image
                    text_to_image_map += [image_index] * captions_per_image
                    image_index += 1

                # B x 5 x 77 -> (B*5) x 77
                texts = torch.flatten(texts, start_dim=0, end_dim=1)
                
                with autocast():
                    if args.distributed and not args.horovod:
                        img_embeds = model.module.encode_image(images)
                    else:
                        img_embeds = model.encode_image(images)
                    img_embeds = F.normalize(img_embeds, dim=-1) # 20 x 1024
                    image_encodings.append(img_embeds)

                    if args.distributed and not args.horovod:
                        text_embeds = model.module.encode_text(texts)
                    else:
                        text_embeds = model.encode_text(texts)
                    text_embeds = F.normalize(text_embeds, dim=-1)
                    text_encodings.append(text_embeds)

            image_encodings = torch.cat(image_encodings)
            text_encodings = torch.cat(text_encodings)

            text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
            image_to_text_map = torch.LongTensor(image_to_text_map).to(device)


            return (image_encodings, text_encodings,
                    text_to_image_map, image_to_text_map)

    def run_zero_shot_COCO(model, preprocess_val, args):
        autocast = get_autocast(args.precision)
        cast_dtype = get_cast_dtype(args.precision)
        tokenizer = get_tokenizer(args.model)
        device = args.device
        coco_root = "/share/data/vdata/coco/images/val2017"
        coco_ann_file = "/share/data/vdata/coco/annotations/captions_val2017.json"
        k_vals = [1,5]
        dataset = CocoCaptions(
            root=coco_root,
            annFile=coco_ann_file,
            transform=preprocess_val,
            # Note: almost all images have 5 captions, but 12/5000 have 6, and 1/5000 has 7 - I ignore these few extra captions.
            target_transform=lambda texts: tokenizer(texts[:5])
        )

        (image_encodings, text_encodings,
                    text_to_image_map, image_to_text_map) = encode_dataset(args, model, dataset)
        with autocast():
            num_text = text_encodings.shape[0]
            num_im = image_encodings.shape[0]
            captions_per_image = image_to_text_map.shape[1]

            print("Text-to-image recall...")

            dist_matrix = text_encodings @ image_encodings.T  # dist_matrix[i] gives logits for ith text

            # Note: this matrix is pretty big (5000 x 25000 with dtype float16 = 250MB)
            #  torch.argsort runs out of memory for me (6GB VRAM) so I move to CPU for sorting
            dist_matrix = dist_matrix.cpu()

            # Sort in descending order; first is the biggest logit
            inds = torch.argsort(dist_matrix, dim=1, descending=True)
            inds = inds.to(device)

            text_to_image_recall = []

            for k in k_vals:
                # Extract top k indices only
                topk = inds[:, :k]

                # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
                correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)

                num_correct = correct.sum().item()
                text_to_image_recall.append(num_correct / num_text)


            # image-to-text recall
            print("Image-to-text recall...")
            dist_matrix = dist_matrix.T  # dist_matrix[i] gives logits for the ith image

            # Sort in descending order; first is the biggest logit
            inds = torch.argsort(dist_matrix, dim=1, descending=True)
            inds = inds.to(device)

            image_to_text_recall = []

            for k in k_vals:
                # Extract top k indices only
                topk = inds[:, :k]

                correct = torch.zeros((num_im,), dtype=torch.bool).cuda()

                #  For each image, check whether one of the 5 relevant captions was retrieved
                # Check if image matches its ith caption (for i=0..4)
                for i in range(captions_per_image):
                    contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
                    correct = torch.logical_or(correct, contains_index)

                
                # find the index of the correct image, ie. where the index is non-zero in correct
                correct_index = torch.nonzero(correct).squeeze(1)

                # save the correct_index as a pickle file
                save_path = "/share/data/pals/kevin/open_clip/src/kevin_3x3_comparisons"
                with open(save_path + f"/correct_index_i2t_top_{k}.pkl", "wb") as f:
                    pickle.dump(correct_index, f)

                num_correct = correct.sum().item()
                image_to_text_recall.append(num_correct / num_im)#

            print("Done.")
            return text_to_image_recall, image_to_text_recall

if PROBABILISTIC_CLIP == True:
    def run(model, text_embeds, dataloader, args):
        autocast = get_autocast(args.precision)
        cast_dtype = get_cast_dtype(args.precision)
        with torch.no_grad():
            top1_p_img, top5_p_img, n_p_img = 0., 0., 0.
            top1_p_txt, top5_p_txt, n_p_txt = 0., 0., 0.
            text_embeds_transposed = text_embeds.T
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

                    # print("shape of img_embeds: ", img_embeds.shape)
                    # print("shape of text_embeds: ", text_embeds.shape)
                    # find log probs
                    device = args.device
                    mu_img, sigma_img = model.module.mu_sigma_img(text_embeds_transposed, device)
                    mu_txt, sigma_txt = model.module.mu_sigma_txt(img_embeds, device)
                    # img_embeds --> 150 x 1024 || mu_img, sigma_img --> 1000 x 1024
                    # text_embeds_transposed --> 1000 x 1024 || mu_txt, sigma_txt --> 150 x 1024

                    # Edit 9/26: adjusting for different shaped image and text embeddings: img--> 150 x 1024, text(after .T)--> 1000 x 1024
                    
                    P_img = calculate_P_lessMemory(img_embeds, mu_img, sigma_img) # 150 x 1000

                    P_txt = calculate_P_lessMemory(text_embeds_transposed, mu_txt, sigma_txt) # 1000 x 150
                    P_txt = P_txt.T # 150 x 1000

                    """
                    P_img is doing image retrieval. Topk of acc is applied for each column where in the case of P_img, each column
                    is the form of P(img_1|txt_1) ... P(img_1|txt__1000). For each column, the givens are iterated over the rows
                    """
                    # P_txt = calculate_P_lessMemory(text_embeds_transposed, mu_txt, sigma_txt) # 150 x 1000
                    """
                    P_txt is doing text retrieval. Topk of acc is applied for each column where in the case of P_txt, each column
                    is the form of P(txt_1|img_1) ... P(txt_1|img_150). For each column, the givens are iterated over the rows.
                    Hence, if we transpose it, we have for each column, the form of P(img_1|txt_1) ... P(img_1|txt__1000)
                    """
                    """
                    Therefore, this code is currently wrong as P_img's columns iterates over the givens(txts).
                    It will be accurate to use P_txt such that the image is the given and is iterated over the columns.
                    """

                    # log_probs = P_img.T + P_txt # 150 x 1000 --> each row == for each image, log prob for each class
                    # print("shape of P_img: ", P_img.shape)
                    # log_probs_p = P_txt

                ###
                # measure accuracy
                acc1_p_img, acc5_p_img = accuracy(P_img, target, topk=(1, 5))
                top1_p_img += acc1_p_img
                top5_p_img += acc5_p_img
                n_p_img += images.size(0)

                acc1_p_txt, acc5_p_txt = accuracy(P_txt, target, topk=(1, 5))
                top1_p_txt += acc1_p_txt
                top5_p_txt += acc5_p_txt
                n_p_txt += images.size(0)

        top1_p_img = (top1_p_img / n_p_img)
        top5_p_img = (top5_p_img / n_p_img)
        top1_p_txt = (top1_p_txt / n_p_txt)
        top5_p_txt = (top5_p_txt / n_p_txt)
        print("######## Zero Shot Results ########")
        print("top1_p_img: ", top1_p_img)
        print("top5_p_img: ", top5_p_img)
        print("top1_p_txt: ", top1_p_txt)
        print("top5_p_txt: ", top5_p_txt)
        return top1_p_img, top5_p_img, top1_p_txt, top5_p_txt
else:
    def run(model, classifier, dataloader, args):
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
                        image_features = model.module.encode_image(images)
                    else:
                        image_features = model.encode_image(images)
                    image_features = F.normalize(image_features, dim=-1)
                    logits = 100. * image_features @ classifier

                # measure accuracy
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)

        top1 = (top1 / n)
        top5 = (top5 / n)
        return top1, top5

def zero_shot_eval(model, data, epoch, args):
    # if 'imagenet-val' not in data and 'imagenet-v2' not in data:
    #     return {}
    # if args.zeroshot_frequency == 0:
    #     return {}
    # if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
    #     return {}

    logging.info('Starting zero-shot imagenet.')

    logging.info('Building zero-shot classifier')
    text_embeds = zero_shot_classifier(model, imagenet_classnames, openai_imagenet_template, args)

    logging.info('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        if PROBABILISTIC_CLIP == True:
            top1_p_img, top5_p_img, top1_p_txt, top5_p_txt = run(model, text_embeds, data['imagenet-val'].dataloader, args)
            results['imagenet-zeroshot-val-top1-p_img'] = top1_p_img
            results['imagenet-zeroshot-val-top5-p_img'] = top5_p_img
            results['imagenet-zeroshot-val-top1-p_txt'] = top1_p_txt
            results['imagenet-zeroshot-val-top5-p_txt'] = top5_p_txt
        else:
            top1, top5 = run(model, text_embeds, data['imagenet-val'].dataloader, args)
            print("Imagenet top1: ", top1)
            print("Imagenet top5: ", top5)
            results['imagenet-zeroshot-val-top1'] = top1
            results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, text_embeds, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5

    if 'coco-val-preprocess_val' in data: 
        preprocess_val = data['coco-val-preprocess_val']

        if PROBABILISTIC_CLIP == True:
            ###### PROBABILISTIC CLIP ############################
            # greg_comp_data(args, model, preprocess_val)
            # greg_comp_probabilistic_clip(args, model, preprocess_val)

            (text_to_image_recall_over_inputs,
            text_to_image_recall_over_givens, 
            image_to_text_recall_over_inputs, 
            image_to_text_recall_over_givens)=run_zero_shot_COCO(model, preprocess_val, args)

            results['coco-zeroshot-val-TOP1-text_to_image_recall_over_inputs'] = text_to_image_recall_over_inputs[0]
            results['coco-zeroshot-val-TOP5-text_to_image_recall_over_inputs'] = text_to_image_recall_over_inputs[1]
            results['coco-zeroshot-val-TOP1-text_to_image_recall_over_givens'] = text_to_image_recall_over_givens[0]
            results['coco-zeroshot-val-TOP5-text_to_image_recall_over_givens'] = text_to_image_recall_over_givens[1]
            results['coco-zeroshot-val-TOP1-image_to_text_recall_over_inputs'] = image_to_text_recall_over_inputs[0]
            results['coco-zeroshot-val-TOP5-image_to_text_recall_over_inputs'] = image_to_text_recall_over_inputs[1]
            results['coco-zeroshot-val-TOP1-image_to_text_recall_over_givens'] = image_to_text_recall_over_givens[0]
            results['coco-zeroshot-val-TOP5-image_to_text_recall_over_givens'] = image_to_text_recall_over_givens[1]
        else: 
            ######## ORIGINAL CLIP ############################
            t2i, i2t = run_zero_shot_COCO(model, preprocess_val, args)
            k_vals = [1, 5]
            print("Text-to-image Recall@K")
            for k, x in zip(k_vals, t2i):
                print(f" R@{k}: {100*x:.2f}%")

            print("Image-to-text Recall@K")
            for k, x in zip(k_vals, i2t):
                print(f" R@{k}: {100*x:.2f}%")
            # greg_comp_data(args, model, preprocess_val)
            
            results['coco-zeroshot-val-TOP1-text_to_image_recall'] = t2i[0]
            results['coco-zeroshot-val-TOP5-text_to_image_recall'] = t2i[1]
            results['coco-zeroshot-val-TOP1-image_to_text_recall'] = i2t[0]
            results['coco-zeroshot-val-TOP5-image_to_text_recall'] = i2t[1]

            #################################################


    logging.info('Finished zero-shot evaluations.')

    return results


# def run_zero_shot_COCO(model, dataset, args):
#     # def P_matrix_topk_acc(P):
#     #     # Get the indices of the sorted values (in descending order) for each row
#     #     sorted_indices = torch.argsort(P, dim=1, descending=False).to(args.device)

#     #     # Get the indices of the main diagonal values
#     #     diagonal_indices = torch.arange(0, P.size(0)).to(args.device)
#     #     """
#     #     In the case of P_img, each row would be P(img_1|txt_1) ... P(img_n|txt_1). 
#     #     Hence, by applying on each row, we are doing image retrieval for given text.

#     #     In the case of P_txt, each row would be P(txt_1|img_1) ... P(txt_n|img_1).
#     #     Hence, by applying on each row, we are doing text retrieval for given image.
#     #     """
#     #     # Check if the index of the main diagonal value is in the top 1 or 5 indices for each row
#     #     top1_mask = sorted_indices[:, 0] == diagonal_indices
#     #     top5_mask = (sorted_indices[:, :5] == diagonal_indices.unsqueeze(1)).any(dim=1)
#     #     return top1_mask, top5_mask
#     # def P_matrix_topk_acc(P, is_text_retrieval=False):
#     #     # Get the indices of the sorted values (in descending order) for each row
#     #     sorted_indices = torch.argsort(P, dim=1, descending=False).to(args.device)

#     #     # Get the indices of the main diagonal values
#     #     diagonal_indices = torch.arange(0, P.size(0)).to(args.device)

#     #     if is_text_retrieval:
#     #         # For text retrieval, check the quotient of sorted_indices and diagonal_indices
#     #         quotient_sorted_indices = sorted_indices // 5
#     #         quotient_diagonal_indices = diagonal_indices // 5

#     #         # Check if the quotient of the index of the main diagonal value is in the top 1 or 5 indices for each row
#     #         top1_mask = quotient_sorted_indices[:, 0] == quotient_diagonal_indices
#     #         top5_mask = (quotient_sorted_indices[:, :5] == quotient_diagonal_indices.unsqueeze(1)).any(dim=1)
#     #     else:
#     #         """
#     #         In the case of P_img, each row would be P(img_1|txt_1) ... P(img_n|txt_1). 
#     #         Hence, by applying on each row, we are doing image retrieval for given text.

#     #         In the case of P_txt, each row would be P(txt_1|img_1) ... P(txt_n|img_1).
#     #         Hence, by applying on each row, we are doing text retrieval for given image.
#     #         """
#     #         # Check if the index of the main diagonal value is in the top 1 or 5 indices for each row
#     #         top1_mask = sorted_indices[:, 0] == diagonal_indices
#     #         top5_mask = (sorted_indices[:, :5] == diagonal_indices.unsqueeze(1)).any(dim=1)

#     #     return top1_mask, top5_mask

#     def P_matrix_topk_acc(P,is_text_retrieval=False):
#         """
#         In the case of P_img, each row would be P(img_1|txt_1) ... P(img_n|txt_1). 
#         Hence, by applying on each row, we are doing image retrieval for given text.
#         P_img --> 100 x 20 

#         In the case of P_txt, each row would be P(txt_1|img_1) ... P(txt_n|img_1).
#         Hence, by applying on each row, we are doing text retrieval for given image.
#         P_txt --> 20 x 100
#         """
#         # sort each row in ascending order of negative log probs 
#         sorted_indices = torch.argsort(P, dim=1, descending=False).to(args.device)

#         if is_text_retrieval:
#             # 20 x 100
#             sorted_indices = sorted_indices // 5
#             correct_image_indices = torch.arange(0, P.size(0)).to(args.device)

#             top1_mask = sorted_indices[:, 0] == correct_image_indices
#             top5_mask = (sorted_indices[:, :5] == correct_image_indices.unsqueeze(1)).any(dim=1)

#         else:
#             # 100 x 20
#             correct_image_indices = torch.arange(0, P.size(0)).to(args.device)
#             correct_image_indices = correct_image_indices // 5

#             top1_mask = sorted_indices[:, 0] == correct_image_indices
#             top5_mask = (sorted_indices[:, :5] == correct_image_indices.unsqueeze(1)).any(dim=1)

#         return top1_mask, top5_mask


#     autocast = get_autocast(args.precision)
#     cast_dtype = get_cast_dtype(args.precision)
#     tokenizer = get_tokenizer(args.model)

#     num_samples = len(dataset)
#     sampler = None
#     shuffle = None

#     dataloader = DataLoader(
#         dataset,
#         batch_size=500,
#         shuffle=shuffle,
#         num_workers=args.workers,
#         pin_memory=True,
#         sampler=sampler,
#     )
#     dataloader.num_samples = num_samples
#     dataloader.num_batches = len(dataloader)

#     top1_mask_list_img_retrieval = []
#     top5_mask_list_img_retrieval = []

#     top1_mask_list_txt_retrieval = []
#     top5_mask_list_txt_retrieval = []

#     with torch.no_grad():
#         for images, texts in dataloader:
            
#             # only take every 5 as they are identical 
#             images = images[::5]

#             # image embeddings
#             images = images.to(args.device) # for batch of 100, 20 images
#             if cast_dtype is not None:
#                 images = images.to(dtype=cast_dtype)
            
#             texts = texts.to(args.device) # for batch of 100, 100 texts 
#             if cast_dtype is not None:
#                 texts = texts.to(dtype=cast_dtype)
                            
#             # text embeddings
#             # texts = tokenizer(texts).to(args.device) # textbreak = text.find("\n", pos) + 1 AttributeError: 'Tensor' object has no attribute 'find'

#             with autocast():
#                 if args.distributed and not args.horovod:
#                     img_embeds = model.module.encode_image(images)
#                 else:
#                     img_embeds = model.encode_image(images)
#                 img_embeds = F.normalize(img_embeds, dim=-1) # 20 x 1024

#                 if args.distributed and not args.horovod:
#                     text_embeds = model.module.encode_text(texts)
#                 else:
#                     text_embeds = model.encode_text(texts)

#                 text_embeds = F.normalize(text_embeds, dim=-1) # 100 x 1024
                
#                 device = args.device

#                 mu_img, sigma_img = model.module.mu_sigma_img(text_embeds, device) # 100 x 1024
#                 P_img = calculate_P_lessMemory(img_embeds, mu_img, sigma_img) # 100 x 20

#                 mu_txt, sigma_txt = model.module.mu_sigma_img(img_embeds, device) # 20 x 1024
#                 P_txt = calculate_P_lessMemory(text_embeds, mu_txt, sigma_txt) # 20 x 100

#                 # image retrieval for given text
#                 top1_mask_P_img = P_matrix_topk_acc(P_img,is_text_retrieval=False)[0] 
#                 top5_mask_P_img = P_matrix_topk_acc(P_img,is_text_retrieval=False)[1]
#                 top1_mask_list_img_retrieval.append(top1_mask_P_img)
#                 top5_mask_list_img_retrieval.append(top5_mask_P_img)

#                 # text retrieval for given image
#                 top1_mask_P_txt = P_matrix_topk_acc(P_txt,is_text_retrieval=True)[0]
#                 top5_mask_P_txt = P_matrix_topk_acc(P_txt,is_text_retrieval=True)[1]
#                 top1_mask_list_txt_retrieval.append(top1_mask_P_txt)
#                 top5_mask_list_txt_retrieval.append(top5_mask_P_txt)
#                 # Compute the Top-1 and Top-5 accuracy
#         top1_mask_img_retrieval = torch.cat(top1_mask_list_img_retrieval, dim=0)
#         top5_mask_img_retrieval = torch.cat(top5_mask_list_img_retrieval, dim=0)

#         top1_mask_txt_retrieval = torch.cat(top1_mask_list_txt_retrieval, dim=0)
#         top5_mask_txt_retrieval = torch.cat(top5_mask_list_txt_retrieval, dim=0)

#         top1_accuracy_img_retrieval = top1_mask_img_retrieval.float().mean().item()
#         top5_accuracy_img_retrieval = top5_mask_img_retrieval.float().mean().item()

#         top1_accuracy_txt_retrieval = top1_mask_txt_retrieval.float().mean().item()
#         top5_accuracy_txt_retrieval = top5_mask_txt_retrieval.float().mean().item()

#     return top1_accuracy_img_retrieval, top5_accuracy_img_retrieval, top1_accuracy_txt_retrieval, top5_accuracy_txt_retrieval
