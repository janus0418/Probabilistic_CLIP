import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        mu_img,
        sigma_img,
        mu_txt,
        sigma_txt,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
            all_mu_img = hvd.allgather(mu_img)
            all_sigma_img = hvd.allgather(sigma_img)
            all_mu_txt = hvd.allgather(mu_txt)
            all_sigma_txt = hvd.allgather(sigma_txt)

        else:
            print("############Line 43 activated############")
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
                all_mu_img = hvd.allgather(mu_img)
                all_sigma_img = hvd.allgather(sigma_img)
                all_mu_txt = hvd.allgather(mu_txt)
                all_sigma_txt = hvd.allgather(sigma_txt)
            
            if not local_loss:
                print("############Line 53 activated############")
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))

                gathered_mu_img = list(all_mu_img.chunk(world_size, dim=0))
                gathered_sigma_img = list(all_sigma_img.chunk(world_size, dim=0))
                gathered_mu_txt = list(all_mu_txt.chunk(world_size, dim=0))
                gathered_sigma_txt = list(all_sigma_txt.chunk(world_size, dim=0))
    
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features

                gathered_mu_img[rank] = mu_img
                gathered_sigma_img[rank] = sigma_img
                gathered_mu_txt[rank] = mu_txt
                gathered_sigma_txt[rank] = sigma_txt

                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)

                all_mu_img = torch.cat(gathered_mu_img, dim=0)
                all_sigma_img = torch.cat(gathered_sigma_img, dim=0)
                all_mu_txt = torch.cat(gathered_mu_txt, dim=0)
                all_sigma_txt = torch.cat(gathered_sigma_txt, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
            all_mu_img = torch.cat(torch.distributed.nn.all_gather(mu_img), dim=0)
            all_sigma_img = torch.cat(torch.distributed.nn.all_gather(sigma_img), dim=0)
            all_mu_txt = torch.cat(torch.distributed.nn.all_gather(mu_txt), dim=0)
            all_sigma_txt = torch.cat(torch.distributed.nn.all_gather(sigma_txt), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            gathered_mu_img = [torch.zeros_like(mu_img) for _ in range(world_size)]
            gathered_sigma_img = [torch.zeros_like(sigma_img) for _ in range(world_size)]
            gathered_mu_txt = [torch.zeros_like(mu_txt) for _ in range(world_size)]
            gathered_sigma_txt = [torch.zeros_like(sigma_txt) for _ in range(world_size)]

            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            dist.all_gather(gathered_mu_img, mu_img)
            dist.all_gather(gathered_sigma_img, sigma_img)
            dist.all_gather(gathered_mu_txt, mu_txt)
            dist.all_gather(gathered_sigma_txt, sigma_txt)

            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                gathered_mu_img[rank] = mu_img
                gathered_sigma_img[rank] = sigma_img
                gathered_mu_txt[rank] = mu_txt
                gathered_sigma_txt[rank] = sigma_txt

            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)
            all_mu_img = torch.cat(gathered_mu_img, dim=0)
            all_sigma_img = torch.cat(gathered_sigma_img, dim=0)
            all_mu_txt = torch.cat(gathered_mu_txt, dim=0)
            all_sigma_txt = torch.cat(gathered_sigma_txt, dim=0)

    return all_image_features, all_text_features, all_mu_img, all_sigma_img, all_mu_txt, all_sigma_txt

# # ORIGINAL VERSION 
class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.pi = 3.14159265358979323846
        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        
        # # Hooks for Checking Error
        # def hook_fn(module, input, output):
        #     name = ""
        #     if not isinstance(output, tuple):  # Ensure output is a tuple
        #         output = (output,)
        #     if not isinstance(input, tuple):  # Ensure output is a tuple
        #         input = (input,)

        #     for idx, tuple_val in enumerate(input):
        #         if torch.distributed.get_rank() == 0:
        #             # if tuple_val is not None:
        #             if tuple_val is not None and torch.isnan(tuple_val).any():
        #                 min = tuple_val.min()
        #                 max = tuple_val.max() 
        #                 print("**** 156 LOSS INPUT NaN value detected in gradients. Stopping training."+f"{module} of {name} contains NaN values at input index {idx}!"+f"Min: {min}, Max: {max}")
        #                 raise ValueError("NAN DETECTED 475")

        #     for idx, tuple_val in enumerate(output):
        #         if torch.distributed.get_rank() == 0:
        #             #if tuple_val is not None:
        #             if tuple_val is not None and torch.isnan(tuple_val).any():
        #                 min = tuple_val.min()
        #                 max = tuple_val.max() 
        #                 print("**** 163 LOSS OUTPUT NaN value detected in gradients. Stopping training."+f"{module} of {name} contains NaN values at input index {idx}!"+f"Min: {min}, Max: {max}")                            
        #                 raise ValueError("NAN DETECTED 482")
        #         # if tuple_val is not None and torch.isnan(tuple_val).any():
        #             # print("#### model 198 OUTPUT NaN value detected in gradients. Stopping training."+f"{module} of {name} contains NaN values at output index {idx}!")
        #             # for idx, tuple_val_input in enumerate(input):
        #             #     if tuple_val_input is not None:
        #             #     # if tuple_val_input is not None and torch.isnan(tuple_val).any():
        #             #         min = tuple_val_input.min()
        #             #         max = tuple_val_input.max() 
        #             #         print("**** 212 INPUT NaN value detected in gradients. Stopping training."+f"{module} of {name} contains NaN values at input index {idx}!"+f"Min: {min}, Max: {max}")
        #             # raise ValueError(f"####### model 204; NaN value detected in gradients Stopping training.")

        #     # for idx, tuple_val in enumerate(input):
        #     #     tuple_val.fill_(0.01)

        # self.register_backward_hook(hook_fn)
    def calculate_P_lessMemory_original(self,Input,Mu,Sigma,device,b,d):
        # sigma here is the reciprocal of the variance sigma = sigma^-1
        Q = torch.empty(0, b).to(device)

        for i in range(Mu.shape[0]):
            Q = torch.cat((Q, torch.sum(torch.square(Input - Mu[i])*Sigma[i], dim=1).reshape(1,b)), dim=0)

        # print(f"Sigma_min: {torch.min(Sigma).item()} || Sigma_max: {torch.max(Sigma).item()}")

        return (Q - torch.sum(torch.log(Sigma), dim=1).reshape(-1,1) + d*torch.log(torch.tensor(2*self.pi).to(device)))/2

    def calculate_P_lessMemory(self,input,mu,sigma,device,b,d):
        # Input = batch x 1024(dimension)
        # mu = batch x 1024(dimension)
        # Sigma = batch x 1024(dimension)

        # here sigma isn't a reciprocal
        mu = mu / (sigma + 0.0000001)
        input = input / (sigma + 0.0000001)

        mu = mu.unsqueeze(0) # [1 x batch x dimension]
        input = input.unsqueeze(0) # [1 x batch x dimension]

        # compute cdist between mu and input
        cdist = torch.cdist(mu, input, p=2.0) # [1 x batch x batch]
        cdist = cdist.squeeze(0) # [batch x batch]
        Q = torch.pow(cdist,2) # [batch x batch]

        # 2nd element

        # check the min and max of sigma
        # define sigma reciprocal
        # print("##### 191 sigma |||| "f"sigma_min: {torch.min(sigma).item()} || sigma_max: {torch.max(sigma).item()}")
        log_sigma = torch.log(sigma)
        log_sigma = torch.sum(log_sigma, dim=1).reshape(-1,1) # [batch x 1]
        # 3rd element; constant is not needed, shared by all loss

        # print out the min and max of Q and log_sigma
        # print("##### 193 Q |||| "f"Q_min: {torch.min(Q).item()} || Q_max: {torch.max(Q).item()}")
        # print("##### 194 log_sigma |||| "f"log_sigma_min: {torch.min(log_sigma).item()} || log_sigma_max: {torch.max(log_sigma).item()}")
        
        # diag_Q = torch.diag(Q) # [batch x 1]
        # quit()

        return Q + log_sigma

    def calculate_loss_less_original(self,P,device):
        # find the main diagonal of res1
        diag = torch.diag(P).to(device)

        return torch.sum(torch.max(torch.max(P, dim=0).values - diag, torch.tensor(0.0))) \
            + torch.sum(torch.max(torch.max(P, dim=1).values - diag, torch.tensor(0.0)))
    
    def calculate_loss_less(self,P,device):
        # find the main diagonal of res1
        diag = torch.diag(P).to(device)

        max_P_dim_0 = torch.max(P, dim=0).values
        max_P_dim_1 = torch.max(P, dim=1).values
    
        minus_diag_dim_0 = max_P_dim_0 - diag
        minus_diag_dim_1 = max_P_dim_1 - diag

        sum_dim_0 = torch.sum(F.relu(minus_diag_dim_0))
        sum_dim_1 = torch.sum(F.relu(minus_diag_dim_1))

        # print min and max of all variables and data type using tensor.datatype
        # print("##### 257 diag |||| "f"diag_min: {torch.min(diag).item()} || diag_max: {torch.max(diag).item()} || diag_dtype: {diag.dtype}")
        # print("##### 258 max_P_dim_0 |||| "f"max_P_dim_0_min: {torch.min(max_P_dim_0).item()} || max_P_dim_0_max: {torch.max(max_P_dim_0).item()} || max_P_dim_0_dtype: {max_P_dim_0.dtype}")
        # print("##### 259 max_P_dim_1 |||| "f"max_P_dim_1_min: {torch.min(max_P_dim_1).item()} || max_P_dim_1_max: {torch.max(max_P_dim_1).item()} || max_P_dim_1_dtype: {max_P_dim_1.dtype}")
        # print("##### 260 minus_diag_dim_0 |||| "f"minus_diag_dim_0_min: {torch.min(minus_diag_dim_0).item()} || minus_diag_dim_0_max: {torch.max(minus_diag_dim_0).item()} || minus_diag_dim_0_dtype: {minus_diag_dim_0.dtype}")
        # print("##### 261 minus_diag_dim_1 |||| "f"minus_diag_dim_1_min: {torch.min(minus_diag_dim_1).item()} || minus_diag_dim_1_max: {torch.max(minus_diag_dim_1).item()} || minus_diag_dim_1_dtype: {minus_diag_dim_1.dtype}")
        # print("##### 262 sum_dim_0 |||| "f"sum_dim_0_min: {torch.min(sum_dim_0).item()} || sum_dim_0_max: {torch.max(sum_dim_0).item()} || sum_dim_0_dtype: {sum_dim_0.dtype}")
        # print("##### 263 sum_dim_1 |||| "f"sum_dim_1_min: {torch.min(sum_dim_1).item()} || sum_dim_1_max: {torch.max(sum_dim_1).item()} || sum_dim_1_dtype: {sum_dim_1.dtype}")

        return sum_dim_0 + sum_dim_1

    def forward(self, image_features, text_features, logit_scale, mu_img, sigma_img, mu_txt, sigma_txt):
        device = image_features.device
        # if self.world_size > 1:
        #     all_image_features, all_text_features = gather_features(
        #         image_features, text_features,
        #         self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

        #     if self.local_loss:
        #         logits_per_image = logit_scale * image_features @ all_text_features.T
        #         logits_per_text = logit_scale * text_features @ all_image_features.T
        #     else:
        #         logits_per_image = logit_scale * all_image_features @ all_text_features.T
        #         logits_per_text = logits_per_image.T
        # else:
        #     logits_per_image = logit_scale * image_features @ text_features.T
        #     logits_per_text = logit_scale * text_features @ image_features.T

        # # calculated ground-truth and cache if enabled
        # num_logits = logits_per_image.shape[0]
        # if self.prev_num_logits != num_logits or device not in self.labels:
        #     labels = torch.arange(num_logits, device=device, dtype=torch.long)
        #     if self.world_size > 1 and self.local_loss:
        #         labels = labels + num_logits * self.rank
        #     if self.cache_labels:
        #         self.labels[device] = labels
        #         self.prev_num_logits = num_logits
        # else:
        #     labels = self.labels[device]

        # total_loss = (
        #     F.cross_entropy(logits_per_image, labels) +
        #     F.cross_entropy(logits_per_text, labels)
        #     ) / 2
        # Find Loss
        
        if 1 == 0:
        # if self.world_size > 1:
            print("############Gather Features Activated############")
            all_image_features, all_text_features, all_mu_img, all_sigma_img, all_mu_txt, all_sigma_txt = gather_features(
                image_features, text_features, mu_img, sigma_img, mu_txt, sigma_txt,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            b = all_image_features.shape[0]
            d = all_image_features.shape[1]

            P_img = self.calculate_P_lessMemory(all_image_features, all_mu_img, all_sigma_img, device,b,d)
            P_txt = self.calculate_P_lessMemory(all_text_features, all_mu_txt, all_sigma_txt, device,b,d)

        else:
            b = image_features.shape[0]
            d = image_features.shape[1]

            P_img = self.calculate_P_lessMemory(image_features, mu_img, sigma_img, device,b,d)
            P_txt = self.calculate_P_lessMemory(text_features, mu_txt, sigma_txt, device,b,d)

            ### EDIT WITH JIAHAO 
            min_P_img = torch.min(P_img).item()
            min_P_txt = torch.min(P_txt).item()
            max_P_img = torch.max(P_img).item()
            max_P_txt = torch.max(P_txt).item()

            # print("#### 250 NEG LOG PROB"+f"min_P_img: {min_P_img} || max_P_img: {max_P_img}")
            # print("#### 251 NEG LOG PROB"+f"min_P_txt: {min_P_txt} || max_P_txt: {max_P_txt}")

        loss_img = self.calculate_loss_less(P_img,device)
        loss_txt = self.calculate_loss_less(P_txt,device)

        # print(f"loss_img: {loss_img} || loss_txt: {loss_txt}")
        # print(f"loss_img_min: {torch.min(loss_img).item()} || loss_img_max: {torch.max(loss_img).item()}")
        # print(f"loss_txt_min: {torch.min(loss_txt).item()} || loss_txt_max: {torch.max(loss_txt).item()}")
        # quit()

        total_loss = (loss_img + loss_txt)
        # print(f"##################  total_loss: {total_loss.item()}")
        return total_loss





# class ClipLoss(nn.Module):
#     def __init__(
#             self,
#             local_loss=False,
#             gather_with_grad=False,
#             cache_labels=False,
#             rank=0,
#             world_size=1,
#             use_horovod=False,
#     ):
#         super().__init__()


#         self.submodule1 = ClipLoss_Submodule(local_loss, gather_with_grad, cache_labels, rank, world_size, use_horovod)
#         self.submodule2 = ClipLoss_Submodule(local_loss, gather_with_grad, cache_labels, rank, world_size, use_horovod)
#         # register hooks to submodule1 and submodule2

#         for name, module in zip(["MODULE 1", "MODULE2"],[self.submodule1,self.submodule2]):
#             def hook_fn(module, input, output):
#                 if not isinstance(output, tuple):  # Ensure output is a tuple
#                     output = (output,)
#                 if not isinstance(input, tuple):  # Ensure output is a tuple
#                     input = (input,)

#                 for idx, tuple_val in enumerate(input):
#                     if torch.distributed.get_rank() == 0:
#                         print("????????????????????????????????????????????????????????????")
#                         if tuple_val is not None:
#                         # if tuple_val is not None and torch.isnan(tuple_val).any():
#                             min = tuple_val.min()
#                             max = tuple_val.max() 
#                             print("!!!!!!!!!!**** 147 SUBMODULE LOSS INPUT NaN value detected in gradients. Stopping training."+f"{module} of {name} contains NaN values at input index {idx}!"+f"Min: {min}, Max: {max}")

#                 for idx, tuple_val in enumerate(output):
#                     if torch.distributed.get_rank() == 0:
#                         if tuple_val is not None:
#                             min = tuple_val.min()
#                             max = tuple_val.max() 
#                             print("!!!!!!!!!!**** 154 SUBMODULE LOSS OUTPUT NaN value detected in gradients. Stopping training."+f"{module} of {name} contains NaN values at input index {idx}!"+f"Min: {min}, Max: {max}")                            
            
#             module.register_backward_hook(hook_fn)


#     def forward(self, image_features, text_features, logit_scale, mu_img, sigma_img, mu_txt, sigma_txt):
#         device = image_features.device
#         b = image_features.shape[0]
#         d = image_features.shape[1]

#         P_img = self.submodule1.calculate_P_lessMemory(image_features, mu_img, sigma_img, device,b,d)
#         P_txt = self.submodule1.calculate_P_lessMemory(text_features, mu_txt, sigma_txt, device,b,d)

#         loss_img = self.submodule2.calculate_loss_less(P_img,device)
#         loss_txt = self.submodule2.calculate_loss_less(P_txt,device)

#         total_loss = (loss_img + loss_txt)

#         return total_loss


# class ClipLoss_Submodule(nn.Module):

#     def __init__(
#             self,
#             local_loss=False,
#             gather_with_grad=False,
#             cache_labels=False,
#             rank=0,
#             world_size=1,
#             use_horovod=False,
#     ):
#         super().__init__()
#         self.local_loss = local_loss
#         self.gather_with_grad = gather_with_grad
#         self.cache_labels = cache_labels
#         self.rank = rank
#         self.world_size = world_size
#         self.use_horovod = use_horovod
#         self.pi = 3.14159265358979323846
#         # cache state
#         self.prev_num_logits = 0
#         self.labels = {}
    

#         # def hook_fn(module, input, output):
#         #     name = ""
#         #     if not isinstance(output, tuple):  # Ensure output is a tuple
#         #         output = (output,)
#         #     if not isinstance(input, tuple):  # Ensure output is a tuple
#         #         input = (input,)

#         #     for idx, tuple_val in enumerate(input):
#         #         if torch.distributed.get_rank() == 0:
#         #             if tuple_val is not None:
#         #             # if tuple_val is not None and torch.isnan(tuple_val).any():
#         #                 min = tuple_val.min()
#         #                 max = tuple_val.max() 
#         #                 print("**** 156 LOSS INPUT NaN value detected in gradients. Stopping training."+f"{module} of {name} contains NaN values at input index {idx}!"+f"Min: {min}, Max: {max}")

#         #     for idx, tuple_val in enumerate(output):
#         #         if torch.distributed.get_rank() == 0:
#         #             if tuple_val is not None:
#         #                 min = tuple_val.min()
#         #                 max = tuple_val.max() 
#         #                 print("**** 163 LOSS OUTPUT NaN value detected in gradients. Stopping training."+f"{module} of {name} contains NaN values at input index {idx}!"+f"Min: {min}, Max: {max}")                            
#         #         # if tuple_val is not None and torch.isnan(tuple_val).any():
#         #             # print("#### model 198 OUTPUT NaN value detected in gradients. Stopping training."+f"{module} of {name} contains NaN values at output index {idx}!")
#         #             # for idx, tuple_val_input in enumerate(input):
#         #             #     if tuple_val_input is not None:
#         #             #     # if tuple_val_input is not None and torch.isnan(tuple_val).any():
#         #             #         min = tuple_val_input.min()
#         #             #         max = tuple_val_input.max() 
#         #             #         print("**** 212 INPUT NaN value detected in gradients. Stopping training."+f"{module} of {name} contains NaN values at input index {idx}!"+f"Min: {min}, Max: {max}")
#         #             # raise ValueError(f"####### model 204; NaN value detected in gradients Stopping training.")

#         #     # for idx, tuple_val in enumerate(input):
#         #     #     tuple_val.fill_(0.01)

#         # self.register_backward_hook(hook_fn)
#     def calculate_P_lessMemory_original(self,Input,Mu,Sigma,device,b,d):
#         # sigma here is the reciprocal of the variance sigma = sigma^-1
#         Q = torch.empty(0, b).to(device)

#         for i in range(Mu.shape[0]):
#             Q = torch.cat((Q, torch.sum(torch.square(Input - Mu[i])*Sigma[i], dim=1).reshape(1,b)), dim=0)

#         # print(f"Sigma_min: {torch.min(Sigma).item()} || Sigma_max: {torch.max(Sigma).item()}")

#         return (Q - torch.sum(torch.log(Sigma), dim=1).reshape(-1,1) + d*torch.log(torch.tensor(2*self.pi).to(device)))/2
    
#     # def calculate_P_lessMemory(self,Input,Mu,Sigma,device,b,d):
#     #     Q = torch.empty(0, b).to(device)
#     #     # Input = batch x 1024(dimension)
#     #     # mu = batch x 1024(dimension)
#     #     # Sigma = batch x 1024(dimension)
#     #     for i in range(Mu.shape[0]):
#     #         Q = torch.cat((Q, torch.sum(torch.square(Input - Mu[i])*Sigma[i], dim=1).reshape(1,b)), dim=0)
        
#     #     constant = d*torch.log(torch.tensor(2*self.pi).to(device))
#     #     q_with_sigma = Q - torch.sum(torch.log(Sigma), dim=1).reshape(-1,1)

#     #     print("##### 163 q_with_sigma |||| "f"q_with_sigma_min: {torch.min(q_with_sigma).item()} || q_with_sigma_max: {torch.max(q_with_sigma).item()}")
#     #     print("##### 163 constant |||| "f"constant_min: {torch.min(constant).item()} || constant_max: {torch.max(constant).item()}")
#     #     quit()
#     #     return (q_with_sigma + constant)/2
    
#     def calculate_P_lessMemory(self,input,mu,sigma,device,b,d):
#         # Input = batch x 1024(dimension)
#         # mu = batch x 1024(dimension)
#         # Sigma = batch x 1024(dimension)

#         # here sigma isn't a reciprocal
#         mu = mu / (sigma + 0.0000001)
#         input = input / (sigma + 0.0000001)

#         mu = mu.unsqueeze(0) # [1 x batch x dimension]
#         input = input.unsqueeze(0) # [1 x batch x dimension]

#         # compute cdist between mu and input
#         cdist = torch.cdist(mu, input, p=2.0) # [1 x batch x batch]
#         cdist = cdist.squeeze(0) # [batch x batch]
#         Q = torch.pow(cdist,2) # [batch x batch]

#         # 2nd element

#         # check the min and max of sigma
#         # define sigma reciprocal
#         # print("##### 191 sigma |||| "f"sigma_min: {torch.min(sigma).item()} || sigma_max: {torch.max(sigma).item()}")
#         log_sigma = torch.log(sigma)
#         log_sigma = torch.sum(log_sigma, dim=1).reshape(-1,1) # [batch x 1]
#         # 3rd element; constant is not needed, shared by all loss

#         # print out the min and max of Q and log_sigma
#         # print("##### 193 Q |||| "f"Q_min: {torch.min(Q).item()} || Q_max: {torch.max(Q).item()}")
#         # print("##### 194 log_sigma |||| "f"log_sigma_min: {torch.min(log_sigma).item()} || log_sigma_max: {torch.max(log_sigma).item()}")
        
#         # diag_Q = torch.diag(Q) # [batch x 1]
#         # quit()

#         return Q + log_sigma



#     # def calculate_P(Input,Mu,Sigma,device,b,d):
#     #     Q = torch.empty(b, 0).to(device)
#     #     for i in range(Mu.shape[1]):
#     #         mu_i = Mu[:,i].reshape(-1,1)
#     #         d_i = Sigma[:,i].reshape(-1,1)
#     #         X = torch.square(Input - mu_i)
#     #         D = X * d_i
#     #         # sum elements in each column of D
#     #         r = torch.sum(D, dim=0)
#     #         # append r to Q
#     #         Q = torch.cat((Q, r.reshape(-1,1)), dim=1)
#     #     # element-wise product of all columns of D

#     #     Sig = torch.log(Sig)

#     #     S_prod_log = torch.sum(Sig, dim=0)



# #     # Element-wise subtraction of S_prod_log from Q
# #     Q = Q - S_prod_log
# #     # Add log(2*pi) to each element of Q
# #     Q = Q + torch.log(torch.tensor(2*pi).to(device))*d
# #     # Divide each element of Q by 2
# #     Q = Q / 2
# #     return Q


#     def calculate_loss_less_original(self,P,device):
#         # find the main diagonal of res1
#         diag = torch.diag(P).to(device)

#         return torch.sum(torch.max(torch.max(P, dim=0).values - diag, torch.tensor(0.0))) \
#             + torch.sum(torch.max(torch.max(P, dim=1).values - diag, torch.tensor(0.0)))
    
#     def calculate_loss_less(self,P,device):
#         # find the main diagonal of res1
#         diag = torch.diag(P).to(device)

#         max_P_dim_0 = torch.max(P, dim=0).values
#         max_P_dim_1 = torch.max(P, dim=1).values
    
#         minus_diag_dim_0 = max_P_dim_0 - diag
#         minus_diag_dim_1 = max_P_dim_1 - diag

#         sum_dim_0 = torch.sum(F.relu(minus_diag_dim_0))
#         sum_dim_1 = torch.sum(F.relu(minus_diag_dim_1))

#         # print min and max of all variables and data type using tensor.datatype
#         # print("##### 257 diag |||| "f"diag_min: {torch.min(diag).item()} || diag_max: {torch.max(diag).item()} || diag_dtype: {diag.dtype}")
#         # print("##### 258 max_P_dim_0 |||| "f"max_P_dim_0_min: {torch.min(max_P_dim_0).item()} || max_P_dim_0_max: {torch.max(max_P_dim_0).item()} || max_P_dim_0_dtype: {max_P_dim_0.dtype}")
#         # print("##### 259 max_P_dim_1 |||| "f"max_P_dim_1_min: {torch.min(max_P_dim_1).item()} || max_P_dim_1_max: {torch.max(max_P_dim_1).item()} || max_P_dim_1_dtype: {max_P_dim_1.dtype}")
#         # print("##### 260 minus_diag_dim_0 |||| "f"minus_diag_dim_0_min: {torch.min(minus_diag_dim_0).item()} || minus_diag_dim_0_max: {torch.max(minus_diag_dim_0).item()} || minus_diag_dim_0_dtype: {minus_diag_dim_0.dtype}")
#         # print("##### 261 minus_diag_dim_1 |||| "f"minus_diag_dim_1_min: {torch.min(minus_diag_dim_1).item()} || minus_diag_dim_1_max: {torch.max(minus_diag_dim_1).item()} || minus_diag_dim_1_dtype: {minus_diag_dim_1.dtype}")
#         # print("##### 262 sum_dim_0 |||| "f"sum_dim_0_min: {torch.min(sum_dim_0).item()} || sum_dim_0_max: {torch.max(sum_dim_0).item()} || sum_dim_0_dtype: {sum_dim_0.dtype}")
#         # print("##### 263 sum_dim_1 |||| "f"sum_dim_1_min: {torch.min(sum_dim_1).item()} || sum_dim_1_max: {torch.max(sum_dim_1).item()} || sum_dim_1_dtype: {sum_dim_1.dtype}")

#         return sum_dim_0 + sum_dim_1

#     def forward(self, image_features, text_features, logit_scale, mu_img, sigma_img, mu_txt, sigma_txt):
#         device = image_features.device
#         # if self.world_size > 1:
#         #     all_image_features, all_text_features = gather_features(
#         #         image_features, text_features,
#         #         self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

#         #     if self.local_loss:
#         #         logits_per_image = logit_scale * image_features @ all_text_features.T
#         #         logits_per_text = logit_scale * text_features @ all_image_features.T
#         #     else:
#         #         logits_per_image = logit_scale * all_image_features @ all_text_features.T
#         #         logits_per_text = logits_per_image.T
#         # else:
#         #     logits_per_image = logit_scale * image_features @ text_features.T
#         #     logits_per_text = logit_scale * text_features @ image_features.T

#         # # calculated ground-truth and cache if enabled
#         # num_logits = logits_per_image.shape[0]
#         # if self.prev_num_logits != num_logits or device not in self.labels:
#         #     labels = torch.arange(num_logits, device=device, dtype=torch.long)
#         #     if self.world_size > 1 and self.local_loss:
#         #         labels = labels + num_logits * self.rank
#         #     if self.cache_labels:
#         #         self.labels[device] = labels
#         #         self.prev_num_logits = num_logits
#         # else:
#         #     labels = self.labels[device]

#         # total_loss = (
#         #     F.cross_entropy(logits_per_image, labels) +
#         #     F.cross_entropy(logits_per_text, labels)
#         #     ) / 2
#         # Find Loss
        
#         if 1 == 0:
#         # if self.world_size > 1:
#             print("############Gather Features Activated############")
#             all_image_features, all_text_features, all_mu_img, all_sigma_img, all_mu_txt, all_sigma_txt = gather_features(
#                 image_features, text_features, mu_img, sigma_img, mu_txt, sigma_txt,
#                 self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
#             b = all_image_features.shape[0]
#             d = all_image_features.shape[1]

#             P_img = self.calculate_P_lessMemory(all_image_features, all_mu_img, all_sigma_img, device,b,d)
#             P_txt = self.calculate_P_lessMemory(all_text_features, all_mu_txt, all_sigma_txt, device,b,d)

#         else:
#             b = image_features.shape[0]
#             d = image_features.shape[1]

#             P_img = self.calculate_P_lessMemory(image_features, mu_img, sigma_img, device,b,d)
#             P_txt = self.calculate_P_lessMemory(text_features, mu_txt, sigma_txt, device,b,d)

#             ### EDIT WITH JIAHAO 
#             min_P_img = torch.min(P_img).item()
#             min_P_txt = torch.min(P_txt).item()
#             max_P_img = torch.max(P_img).item()
#             max_P_txt = torch.max(P_txt).item()

#             # print("#### 250 NEG LOG PROB"+f"min_P_img: {min_P_img} || max_P_img: {max_P_img}")
#             # print("#### 251 NEG LOG PROB"+f"min_P_txt: {min_P_txt} || max_P_txt: {max_P_txt}")

#         loss_img = self.calculate_loss_less(P_img,device)
#         loss_txt = self.calculate_loss_less(P_txt,device)

#         # print(f"loss_img: {loss_img} || loss_txt: {loss_txt}")
#         # print(f"loss_img_min: {torch.min(loss_img).item()} || loss_img_max: {torch.max(loss_img).item()}")
#         # print(f"loss_txt_min: {torch.min(loss_txt).item()} || loss_txt_max: {torch.max(loss_txt).item()}")
#         # quit()

#         total_loss = (loss_img + loss_txt)
#         # print(f"##################  total_loss: {total_loss.item()}")
#         return total_loss
  
