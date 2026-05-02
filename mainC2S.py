import os
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from diffusers import DDIMScheduler,LCMScheduler
from torchvision.utils import save_image
from PIL import Image
from utils.pipeline import ZePoPipeline
from utils.attn_control import AttentionStyle
import utils.ptp_utils as ptp_utils
from datetime import datetime
import torchvision.transforms as transforms


def load_image(image_path, res, device, gray=False):
    image = Image.open(image_path).convert('RGB') if not gray else Image.open(image_path).convert('L')
    image = torch.tensor(np.array(image)).float()
    if gray:
        image = image.unsqueeze(-1).repeat(1,1,3)
    image = image.permute(2, 0, 1)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (res, res))
    image = image.to(device)
    return image

def get_image(data_dir):
    img_list = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if (file.endswith(".jpg")
                or file.endswith(".png")
                or file.endswith(".bmp")
                or file.endswith(".jpeg")
            ):
                img_list.append(os.path.join(root, file))
    assert len(img_list) > 0, "[ERROR] img_list is Empty!"
    return img_list

def load_mask(image_path, res, device):
    if image_path != '':
        image = Image.open(image_path).convert('RGB')
        image = torch.tensor(np.array(image)).float()
        image = image.permute(2, 0, 1)
        image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
        image = F.interpolate(image, (res, res))
        image = image.to(device)
        image = image[:, :1, :, :]
    else:
        return None
    return image

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--start_ac_layer", type=int, default=8) #8
    args.add_argument("--end_ac_layer", type=int, default=16)  #16
    args.add_argument("--res", type=int, default=512)
    args.add_argument("--cfg_guidance", type=float, default=2)
    args.add_argument("--sty_guidance", type=float, default=0.3)
    args.add_argument("--mix_q_scale", type=float, default=0.75)
    args.add_argument("--prompt", type=str, default='face')
    args.add_argument("--neg_prompt", type=str, default='')
    args.add_argument("--output", type=str, default='./results/')
    args.add_argument("--content", type=str, default='data/facecnt')
    args.add_argument("--style", type=str, default='data/facesty')
    args.add_argument('--content_img_folder_maskh', type=str,
                      default="data\cnt_maskh", help='content image paths')
    args.add_argument('--content_img_folder_maskf', type=str,
                      default="data\cnt_maskf", help='content image paths')
    args.add_argument('--style_img_folder_mask', type=str,
                      default="data/sty_masks", help='style image paths')
    args.add_argument("--model_path", type=str, default='lcm')
    args.add_argument("--num_inference_steps", type=int, default=4)
    args.add_argument("--fix_step_index", type=int, default=99)
    args.add_argument("--tome", action="store_true")
    args.add_argument("--tome_ratio", type=float, default=0.5)
    args.add_argument("--resizes", type=int, default=512)
    args = args.parse_args()

    out_dir = args.output
    start_ac_layer = args.start_ac_layer
    end_ac_layer = args.end_ac_layer
    num_inference_steps = args.num_inference_steps
    sty_guidance = args.sty_guidance
    fix_step_index = args.fix_step_index
    mix_q_scale = args.mix_q_scale
    de_bug = False
    tome = args.tome
    tome_sx=2
    tome_sy=2
    tome_ratio=args.tome_ratio
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    cfg_scale = args.cfg_guidance
    source_prompt = [args.prompt]
    model_path = args.model_path
    model = ZePoPipeline.from_pretrained(model_path).to(device)
    # 检查并加载缺少的组件
    if not hasattr(model, 'scheduler'):
        model.scheduler = LCMScheduler.from_config(model.scheduler.config)
        print('load success!')
    prompts = source_prompt * 3
    neg_prompt = ''
    Controll_factor = 0.5
    # read  image floder ====================================================
    con_list = get_image(args.content)
    sty_list = get_image(args.style)
    con_list_maskh = get_image(args.content_img_folder_maskh)
    con_list_maskf = get_image(args.content_img_folder_maskf)
    sty_list_mask = get_image(args.style_img_folder_mask)
    con_list += con_list_maskh
    con_list += con_list_maskf
    sty_list += sty_list_mask

    lens = int(len(con_list) / 3)
    print(len)
    for i, content in enumerate(con_list[:lens]):
        source = load_image(content, args.res, device)
#       mask merge
        source_mask_h = load_mask(con_list[i + lens], res=args.resizes, device=device)
        source_mask_f = load_mask(con_list[i + lens * 2], res=args.resizes, device=device)
        source_mask_f = torch.where(source_mask_f ==-1, torch.tensor(0), source_mask_f)
        source_mask_h = torch.where(source_mask_h == -1, torch.tensor(0), source_mask_h)
        # source_mask = (source_mask_f * Controll_factor + source_mask_h * (1-Controll_factor))
        # print(source_mask_f)
        # print(source_mask_h)
        for j, style in enumerate(sty_list[:lens]):
            style = load_image(style, args.resizes, device)
            # mask merge
            style_mask = load_mask(sty_list[j + lens], res=64, device=device)
            style_mask = torch.where(style_mask == -1, torch.tensor(0), style_mask)
            # print(style_mask)
            controller = AttentionStyle(num_inference_steps, start_ac_layer, end_ac_layer,style_guidance=sty_guidance, mix_q_scale=mix_q_scale,
                                        de_bug=de_bug, style_mask=style_mask, source_mask_h=source_mask_h, source_mask_f=source_mask_f )

            ptp_utils.register_attention_control(model, controller, tome, sx=tome_sx,
                                                 sy=tome_sy, ratio=tome_ratio, de_bug=de_bug,)
            time_begin = datetime.now()
            with torch.no_grad():
                torch.cuda.empty_cache()
                generate_image = model(prompt=prompts,negative_prompt=neg_prompt,
                                       image=source, style=style, num_inference_steps=num_inference_steps,
                                       eta=0.0, guidance_scale=cfg_scale,
                                       strength=0.5, save_intermediate=False,
                                       fix_step_index=fix_step_index,
                                       de_bug=de_bug,callback = None,style_mask=style_mask, source_mask_h=source_mask_h, source_mask_f=source_mask_f).images
                os.makedirs(out_dir, exist_ok=True)
                generate_image = torch.from_numpy(generate_image).permute(0, 3, 1, 2)
                # save_image(generate_image, save_name, nrow=3, padding=0)
                save_image(generate_image[-1:], os.path.join(out_dir, f"{i}_{j}.jpg"), nrow=1, padding=0)
            time_end = datetime.now()
            print(f"Finish processing Time cost: {time_end-time_begin}, \nPer image cost: {(time_end-time_begin)/len(os.listdir(args.content))}")

if __name__ == "__main__":
    main()

    # pip install --upgrade diffusers transformers  huggingface_hub -i http://mirrors.aliyun.com/pypi/simple/  --trusted-host mirrors.aliyun.com




