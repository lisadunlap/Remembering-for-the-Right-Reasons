

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from torchray.attribution.grad_cam import grad_cam as tr_gradcam

import matplotlib
matplotlib.rcParams.update({'font.size': 18})
import cv2
import matplotlib.pyplot as plt

def deeplabv3_attention(model, transform, file_path, index, device,
                        plot_vis=False, save_vis_path=None):
    img_vis = Image.open(file_path).convert('RGB')
    img = transform(img_vis).unsqueeze(0).to(device)

    # Model returns dict:
    # - 'out': tensor shape B x 21 x H x W
    # - 'aux': tensor shape B x 21 x H x W
    out = model(img)
    maxed = out['out'][0].argmax(0)

    # mask: shape 1 x H x W
    mask = (maxed == index).float().cpu().unsqueeze(0).unsqueeze(0)

    if plot_vis:
        fig, ax = plt.subplots(1,2, figsize=(6,10))
        ax[0].imshow(img_vis)
        ax[1].imshow(mask[0][0].numpy(), cmap='gray')
        ax[0].axis('off')
        ax[1].axis('off')
        plt.tight_layout()
        if save_vis_path is not None:
            plt.savefig(save_vis_path, bbox_inches='tight')
        else:
            plt.show()

    return {
        'mask': mask
    }



def transformer_attention(model, preprocess, file_path, text_list, tokenized_text, device,
                          plot_vis=False, save_vis_path=None, resize=False):
    image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)

    logits_per_image, _ = model(image, tokenized_text)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

    attentions = []
    unnormalized_attentions = []
    for idx in range(len(text_list)):
        one_hot = np.zeros((1, logits_per_image.size()[-1]), dtype=np.float32)
        one_hot[0, idx] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(device) * logits_per_image)
        model.zero_grad()
        one_hot.backward(retain_graph=True)

        image_attn_blocks = list(
            dict(model.visual.transformer.resblocks.named_children()).values()
        )
        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        R = torch.eye(num_tokens, num_tokens,
                      dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
        for blk in image_attn_blocks:
            grad = blk.attn_grad
            cam = blk.attn_probs
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            R += torch.matmul(cam, R)
        R[0, 0] = 0
        image_relevance = R[0, 1:]

        image_relevance = image_relevance.reshape(1, 1, 7, 7)
        image_relevance = image_relevance.detach().type(torch.float32).cpu()
        if resize:
           image_relevance  = F.interpolate(
               image_relevance, image.shape[2:], mode="bilinear", align_corners=False
            )
        unnormalized_attentions.append(image_relevance)
        image_relevance = normalize(image_relevance)
        attentions.append(image_relevance)

    if plot_vis:
        plot_attention_helper(image, attentions, unnormalized_attentions, probs, text_list,
                          save_vis_path=save_vis_path, resize=resize)

    return {
        'unnormalized_attentions': torch.cat(unnormalized_attentions),
        'attentions': torch.cat(attentions),
        'text_list': text_list,
        'probs': probs
    }


def clip_gcam(model, preprocess, file_path, text_list, tokenized_text, layer, device,
              plot_vis=False, save_vis_path=None, resize=False):

    image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)

    image_features = model.encode_image(image)
    text_features = model.encode_text(tokenized_text).detach()
    image_features_norm = image_features.norm(dim=-1, keepdim=True)
    image_features_new = image_features / image_features_norm
    text_features_norm = text_features.norm(dim=-1, keepdim=True)
    text_features_new = text_features / text_features_norm
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features_new @ text_features_new.t()
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

    # Each shape 1 x 1 x H x W
    attentions = []
    unnormalized_attentions = []
    for i in range(len(text_list)):
        # mutliply the normalized text embedding with image norm to get approx image embedding
        text_prediction = (text_features_new[[i]] * image_features_norm)
        saliency = tr_gradcam(model.visual, image.type(model.dtype), text_prediction, saliency_layer=layer)
        saliency = saliency.detach().type(torch.float32).cpu()
        if resize:
            saliency = F.interpolate(
                saliency, image.shape[2:], mode="bilinear", align_corners=False
            )
        unnormalized_attentions.append(saliency)
        sal = normalize(saliency)
        attentions.append(saliency)

    if plot_vis:
        plot_attention_helper(image, attentions, unnormalized_attentions, probs, text_list,
                          save_vis_path=save_vis_path, resize=resize)

    return {
        'unnormalized_attentions': torch.cat(unnormalized_attentions),
        'attentions': torch.cat(attentions),
        'text_list': text_list,
        'probs': probs
    }

def clip_gcam_train(model, preprocess, inputs, text_list, tokenized_text, layer, device,
              plot_vis=False, save_vis_path=None, resize=False):

    # image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
    attentions = []
    unnormalized_attentions = []
    for im in inputs:
        image = im.unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        text_features = model.encode_text(tokenized_text).detach()
        image_features_norm = image_features.norm(dim=-1, keepdim=True)
        image_features_new = image_features / image_features_norm
        text_features_norm = text_features.norm(dim=-1, keepdim=True)
        text_features_new = text_features / text_features_norm
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features_new @ text_features_new.t()
        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

        # Each shape 1 x 1 x H x W
        for i in range(len(text_list)):
            # mutliply the normalized text embedding with image norm to get approx image embedding
            text_prediction = (text_features_new[[i]] * image_features_norm)
            saliency = tr_gradcam(model.visual, image.type(model.dtype), text_prediction, saliency_layer=layer)
            saliency = saliency.detach().type(torch.float32).cpu()
            if resize:
                saliency = F.interpolate(
                    saliency, image.shape[2:], mode="bilinear", align_corners=False
                )
            unnormalized_attentions.append(saliency)
            sal = normalize(saliency)
            attentions.append(saliency)

    if plot_vis:
        plot_attention_helper(image, attentions, unnormalized_attentions, probs, text_list,
                          save_vis_path=save_vis_path, resize=resize)

    return {
        'unnormalized_attentions': torch.cat(unnormalized_attentions),
        'attentions': torch.cat(attentions),
        'text_list': text_list,
        'probs': probs
    }


# def gcam_attention(model, image, text, text_list, target_layer, create_plot=False):

#     image_vis = image[0].permute(1, 2, 0).data.cpu().numpy()
#     image_vis = (image_vis - image_vis.min()) / (image_vis.max() - image_vis.min())

#     all_vis = []
#     rels = []
#     for idx in range(len(text_list)):
#         gcam = GradCAM(model)
#         probs = gcam.forward(image, text) # same for all iters but need new fwd
#         ids = torch.LongTensor([[idx]]).T.to(device)
#         gcam.backward(ids)
#         att_raw = gcam.generate(target_layer=target_layer)
#         att = att_raw[0][0].cpu().numpy()

#         rels.append(att)

#         if create_plot:
#             vis = show_cam_on_image(image_vis, att)
#             vis = np.uint8(255 * vis)
#             all_vis.append(vis)

#     rels = np.stack(rels)
#     rels = np.mean(rels, axis=0).astype('float32')
#     denom = 1 if (rels.max() - rels.min()) == 0 else (rels.max() - rels.min())
#     rels = (rels - rels.min()) / denom

#     if create_plot:
#         plot([probs], all_vis, text_list, np.arange(len(text_list)), image_vis)

#     return rels


def normalize(sal):
    # sal = tensor of shape 1,1,H,W
    B, C, H, W = sal.shape
    sal = sal.view(B, -1)
    sal_max = sal.max(dim=1, keepdim=True)[0]
    sal_max[torch.where(sal_max == 0)] = 1. # prevent divide by 0
    sal -= sal.min(dim=1, keepdim=True)[0]
    #gcam /= gcam.max(dim=1, keepdim=True)[0]
    sal /= sal_max
    sal = sal.view(B, C, H, W)
    return sal


def show_cam_on_image(img, attention):
    heatmap = cv2.applyColorMap(np.uint8(255 * attention), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def plot_attention_helper(image, attentions, unnormalized_attentions, probs, text_list,
                          save_vis_path=None, resize=False):
    image_vis = image[0].permute(1, 2, 0).data.cpu().numpy()
    image_vis = (image_vis - image_vis.min()) / (image_vis.max() - image_vis.min())
    attention_vis = []
    for i in range(len(attentions)):
        if not resize:
            sal = F.interpolate(
                unnormalized_attentions[i],
                image.shape[2:], mode="bilinear", align_corners=False
            )
            sal = normalize(sal)[0][0]
        else:
            sal = attentions[i][0][0]
        vis = show_cam_on_image(image_vis, sal.numpy())
        vis = np.uint8(255 * vis)
        attention_vis.append(vis)
    plot(probs, attention_vis, text_list, image_vis, save_path=save_vis_path)


def plot(probs, attention_vis, text_list, image_vis, save_path=None):
    sort = np.argsort(probs)[0][::-1]
    attention_vis = np.array(attention_vis)[sort]
    probs_vis = probs[0][sort]
    text_list_vis = np.array(text_list)[sort]

    fig, ax = plt.subplots(1,1+len(text_list),figsize=(5*(1+len(text_list)),6))
    ax[0].imshow(image_vis)
    ax[0].axis('off')

    for idx in range(len(text_list)):
        ax[idx+1].imshow(attention_vis[idx])
        ax[idx+1].axis('off')
        split = text_list_vis[idx].split(' ')
        if len(split) > 6:
            prompt = ' '.join(split[:6]) + '\n ' + ' '.join(split[6:])
        else:
            prompt = text_list_vis[idx]
        ax[idx+1].set_title('{}\n{:.3f}'.format(prompt, probs_vis[idx], 2),
                           fontsize=14)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    return


def parse_attention(attentions, shape, mode='average_nonzero'):
    """
    Chooses b/w different modes of combining attentions.
    attentions: torch tensor of shape B x num_attentions_per_sample x 1 x H x W
    shape: tuple of desired attentions shape to resize to if needed

    mode options:
    - "average_nonzero": average only attentions that are not all zeros.
        If all attentions are nonzero, entry in list is None
    - "max"

    Returns tensor of shape num_nonzero_attentions x 1 x H x W,
    and list of length B which is 1 if attentions[i] was not all 0, and 0 if it was.
    """
    #assert mode == 'average_nonzero', 'only average_nonzero mode implemented'
    final_attentions = []
    inds = []
    for idx in range(attentions.shape[0]):
        # bool tensor of which attentions are not all zero
        nonzero_atts = torch.stack([~torch.all(att == 0) for att in attentions[idx]])
        if torch.all(nonzero_atts.float() == 0):
            inds.append(0)
        else:
            valid = attentions[idx][nonzero_atts]
            if mode == 'average_nonzero':
                att = valid.mean(dim=0)
            elif mode == 'max':
                att = torch.max(valid, dim=0, keepdim=True).values[0]
            else:
                print('ERROR: attention parsing mode {} not implemented'.format(mode))
                raise NotImplementedError
            if len(att.shape) == 2:
                att = att.unsqueeze(0).unsqueeze(0)
            elif len(att.shape) == 3:
                att = att.unsqueeze(0)
            else:
                print(att.shape)
                raise NotImplementedError
            normalized = normalize(att)
            normalized = normalized[0][0].unsqueeze(0)
            if normalized.shape[-2:] != shape:
                normalized = transforms.functional.resize(normalized, shape)
            final_attentions.append(normalized)
            inds.append(1)
    if len(final_attentions) > 0:
        final_attentions = torch.stack(final_attentions)
    return final_attentions, torch.Tensor(inds)


def compute_gradcam(fmaps, logits, labels, device, resize=False, resize_shape=None):
    if logits.shape[1] == 1:
        probs = nn.Sigmoid()(logits)
        class1_probs = 1 - probs
        y = torch.cat((class1_probs, probs), dim=1)
    else:
        y = logits
    ids     = torch.LongTensor([labels.tolist()]).T.to(device)
    one_hot = torch.zeros_like(y).to(device)
    one_hot.scatter_(1, ids, 1.0)

    grads = torch.autograd.grad(y,
                                fmaps,
                                grad_outputs=one_hot,
                                retain_graph=True,
                                create_graph=True)[0]

    weights = F.adaptive_avg_pool2d(grads, 1)

    gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
    gcam = F.relu(gcam)

    if resize:
        assert resize_shape is not None
        gcam = F.interpolate(
            gcam, resize_shape, mode="bilinear", align_corners=False
        )

    B, C, H, W = gcam.shape
    gcam = gcam.view(B, -1)
    gcam_max = gcam.max(dim=1, keepdim=True)[0]
    gcam_max[torch.where(gcam_max == 0)] = 1. # prevent divide by 0
    gcam -= gcam.min(dim=1, keepdim=True)[0]
    #gcam /= gcam.max(dim=1, keepdim=True)[0]
    gcam /= gcam_max
    gcam = gcam.view(B, C, H, W)
    return gcam

def get_displ_img(img):
    try:
        img = img.cpu().numpy().transpose((1, 2, 0))
    except:
        img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    displ_img = std * img + mean
    displ_img = np.clip(displ_img, 0, 1)
    displ_img /= np.max(displ_img)
    displ_img = displ_img
    displ_img = np.uint8(displ_img*255)
    return displ_img/np.max(displ_img)
