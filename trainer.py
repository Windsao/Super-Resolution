import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp, teacher=None):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.teacher = teacher
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            if self.args.data_aug == 'cutmix':
                hr, lr = self.cutmix(hr.clone(), lr.clone(), self.args.prob, self.args.aug_alpha)
            elif self.args.data_aug == 'mixup':
                hr, lr = self.mixup(hr.clone(), lr.clone(), self.args.prob, self.args.aug_beta)
            elif self.args.data_aug == 'cutmixup':
                hr, lr = self.cutmixup(hr.clone(), lr.clone(), self.args.prob, self.args.aug_beta, self.args.prob, self.args.aug_alpha)

            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            if epoch < self.args.start_aug:
                sr = self.model(lr, 0)
                loss = self.loss(sr, hr)
            else:
                sr = self.model(lr, 0)
                aug_img = self.attack_pgd(lr, hr, epsilon=self.args.eps, alpha=self.args.alpha, attack_iters=self.args.iters)
                aug_sr = self.model(aug_img, 0)
                loss = (1 - self.args.beta) * self.loss(sr,hr) + self.args.beta * self.loss(aug_sr,hr)
            
            '''
            if epoch <= 100:
                aug_img = self.attack_pgd(lr, hr, epsilon=1, alpha=1, attack_iters=1)
                sr = self.model(aug_img, 0)
            else:
                sr = self.model(lr, 0)
            
            loss = self.loss(sr, hr)
            '''

            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def distillation(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)

        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            if self.args.data_aug == 'cutmix':
                hr, lr = self.cutmix(hr.clone(), lr.clone(), self.args.prob, self.args.aug_alpha)
            elif self.args.data_aug == 'mixup':
                hr, lr = self.mixup(hr.clone(), lr.clone(), self.args.prob, self.args.aug_beta)
            elif self.args.data_aug == 'cutmixup':
                hr, lr = self.cutmixup(hr.clone(), lr.clone(), self.args.prob, self.args.aug_beta, self.args.prob, self.args.aug_alpha)
            elif self.args.data_aug == 'cutout_mask':
                high_mask, low_mask = self.cutout_mask(lr)
            elif self.args.data_aug == 'cutmix_mask':
                hr, lr, high_mask, low_mask = self.cutmix_mask(hr.clone(), lr.clone())
            elif self.args.data_aug == 'saliency_mask':
                lr, hr = self.prepare(lr, hr)
                lr.requires_grad = True
                sr = self.model(lr, 0)
                loss = self.loss(sr, hr)
                grad = torch.autograd.grad(loss, lr)[0]
                hr, lr, high_mask, low_mask = self.saliency_mask(grad, hr.clone(), lr.clone())

            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            # self.drawInOut(lr, hr, 3, 'saliency_Mask_In')

            self.optimizer.zero_grad()
            if self.args.data_aug == 'advcutmix':
                aug_img = self.attack_pgd(lr, hr, epsilon=self.args.eps, alpha=self.args.alpha, attack_iters=self.args.iters)
                hr, lr = self.cutmix(hr.clone(), aug_img.clone(), self.args.prob, self.args.aug_alpha)
                
            t_sr = self.teacher(lr, 0)
            sr = self.model(lr, 0)
            if 'mask' in self.args.data_aug:
                loss = (1 - self.args.beta) * self.loss(sr.mul(high_mask), hr.mul(high_mask)) + self.args.beta * self.loss(sr.mul(high_mask), t_sr.mul(high_mask))
            else:
                loss = (1 - self.args.beta) * self.loss(sr, hr) + self.args.beta * self.loss(sr, t_sr)
            # self.drawOutOut(t_sr.mul(high_mask), hr.mul(high_mask), 3, 'saliency_Mask_Out')
            # exit()
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        psnr_list = []
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)
                    temp = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)
                    psnr_list.append(temp)
              
                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )
        np.save('psnr.npy', np.array(psnr_list))
        exit()
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

    def attack_pgd(self, lr, hr, epsilon, alpha, attack_iters):
        aug_img = lr.clone().detach()
        aug_img = aug_img + torch.empty_like(aug_img).uniform_(-epsilon, epsilon).cuda()
        # delta.data = self.clamp(delta, lower_limit - lr, upper_limit - lr)
        aug_img = torch.clamp(aug_img, min=0, max=255).detach()
        for _ in range(attack_iters):
            aug_img.requires_grad = True
            sr = self.model(aug_img, 0)
            loss = self.loss(sr, hr)
            # loss.backward()
            grad = torch.autograd.grad(loss, aug_img, retain_graph=False, create_graph=False)[0]
            aug_img = aug_img.detach() + alpha * grad.sign()
            delta = torch.clamp(aug_img - lr, min=-epsilon, max=epsilon)
            aug_img = torch.clamp(lr + delta, min=0, max=255).detach()
          
        return aug_img

    def loss_fn_kd(outputs, labels, teacher_outputs, args):
        alpha = args.alpha
        T = args.temperature
        KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + F.cross_entropy(outputs, labels) * (1. - alpha)

        return KD_loss

    def clamp(X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)

    def mixup(self, im1, im2, prob=1.0, alpha=1.2):
        if alpha <= 0 or np.random.rand(1) >= prob:
            return im1, im2

        v = np.random.beta(alpha, alpha)
        r_index = torch.randperm(im1.size(0)).to(im2.device)

        im1 = v * im1 + (1-v) * im1[r_index, :]
        im2 = v * im2 + (1-v) * im2[r_index, :]
        return im1, im2

    def _cutmix(self, im2, prob=1.0, alpha=1.0):
        if alpha <= 0 or np.random.rand(1) >= prob:
            return None

        cut_ratio = np.random.randn() * 0.01 + alpha

        h, w = im2.size(2), im2.size(3)
        ch, cw = np.int(h*cut_ratio), np.int(w*cut_ratio)

        fcy = np.random.randint(0, h-ch+1)
        fcx = np.random.randint(0, w-cw+1)
        tcy, tcx = fcy, fcx
        rindex = torch.randperm(im2.size(0)).to(im2.device)

        return {
            "rindex": rindex, "ch": ch, "cw": cw,
            "tcy": tcy, "tcx": tcx, "fcy": fcy, "fcx": fcx,
        }

    def cutmix(self, im1, im2, prob=1.0, alpha=1.0):
        c = self._cutmix(im2, prob, alpha)
        if c is None:
            return im1, im2

        scale = im1.size(2) // im2.size(2)
        rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
        tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

        hch, hcw = ch*scale, cw*scale
        hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale          
        im2[..., tcy:tcy+ch, tcx:tcx+cw] = im2[rindex, :, fcy:fcy+ch, fcx:fcx+cw]
        im1[..., htcy:htcy+hch, htcx:htcx+hcw] = im1[rindex, :, hfcy:hfcy+hch, hfcx:hfcx+hcw]

        return im1, im2
    
    def cutmixup(
        self, im1, im2,
        mixup_prob=1.0, mixup_alpha=1.0,
        cutmix_prob=1.0, cutmix_alpha=1.0
    ):
        c = self._cutmix(im2, cutmix_prob, cutmix_alpha)
        if c is None:
            return im1, im2 

        scale = im1.size(2) // im2.size(2)
        rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
        tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

        hch, hcw = ch*scale, cw*scale
        hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

        v = np.random.beta(mixup_alpha, mixup_alpha)
        if mixup_alpha <= 0 or np.random.rand(1) >= mixup_prob:
            im2_aug = im2[rindex, :]
            im1_aug = im1[rindex, :]

        else:
            im2_aug = v * im2 + (1-v) * im2[rindex, :]
            im1_aug = v * im1 + (1-v) * im1[rindex, :]

        # apply mixup to inside or outside
        if np.random.random() > 0.5:
            im2[..., tcy:tcy+ch, tcx:tcx+cw] = im2_aug[..., fcy:fcy+ch, fcx:fcx+cw]
            im1[..., htcy:htcy+hch, htcx:htcx+hcw] = im1_aug[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
        else:
            im2_aug[..., tcy:tcy+ch, tcx:tcx+cw] = im2[..., fcy:fcy+ch, fcx:fcx+cw]
            im1_aug[..., htcy:htcy+hch, htcx:htcx+hcw] = im1[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
            im2, im1 = im2_aug, im1_aug

        return im1, im2

    def cutout_mask(self, lr):
        num_img = lr.size(0)
        lrsize = lr.size(-1)
        l_mask = torch.ones_like(lr).cuda()
        for i in range(num_img):
            mask_x = random.randint(0, lrsize - self.args.mask_size)
            mask_y = random.randint(0, lrsize - self.args.mask_size)
            l_mask[i, : , mask_y: mask_y + self.args.mask_size, mask_x: mask_x + self.args.mask_size] = 0
        h_mask = F.interpolate(l_mask, scale_factor=int(self.scale[0]), mode="nearest")
        return h_mask, l_mask

    def cutmix_mask(self, im1, im2):
        h, w = im2.size(2), im2.size(3)
        ch, cw = self.args.mask_size,  self.args.mask_size

        fcy = np.random.randint(0, h-ch+1)
        fcx = np.random.randint(0, w-cw+1)
        tcy, tcx = fcy, fcx
        rindex = torch.randperm(im2.size(0)).to(im2.device)
        
        scale = im1.size(2) // im2.size(2) # hr, lr

        hch, hcw = ch*scale, cw*scale
        hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale          
        im2[..., tcy:tcy+ch, tcx:tcx+cw] = im2[rindex, :, fcy:fcy+ch, fcx:fcx+cw]
        im1[..., htcy:htcy+hch, htcx:htcx+hcw] = im1[rindex, :, hfcy:hfcy+hch, hfcx:hfcx+hcw]
        
        l_mask = torch.ones_like(im2).cuda()
        l_mask[..., tcy:tcy+ch, tcx:tcx+cw] = 0
        h_mask = F.interpolate(l_mask, scale_factor=scale, mode="nearest")

        return im1, im2, h_mask, l_mask

    def saliency_mask(self, grad, im1, im2):
        filter = torch.ones([1, 3, self.args.mask_size, self.args.mask_size]).float().cuda()
        grad = torch.abs(grad)
        mask_grad = F.conv2d(grad, filter, stride=self.args.mask_size)
        mask_grad = mask_grad.view(mask_grad.size(0), -1)
        mask_index = mask_grad.argsort(descending=False)[:, :1]

        l_mask = torch.ones_like(im2).cuda()
        h, w = im2.size(2), im2.size(3)
        patch_num_per_line = h // self.args.mask_size
        for i in range(len(mask_index)):
            mask_x = (mask_index[i] % patch_num_per_line) * self.args.mask_size
            mask_y = (mask_index[i] // patch_num_per_line) * self.args.mask_size
            l_mask[i, : , mask_y: mask_y + self.args.mask_size, mask_x: mask_x + self.args.mask_size] = 0
        h_mask = F.interpolate(l_mask, scale_factor=int(self.scale[0]), mode="nearest")    
         
        return im1.mul(h_mask), im2.mul(l_mask), h_mask, l_mask

    def drawInOut(self, im1, im2, index, name):
        p1 = im1[index].detach().cpu().permute(1,2,0).numpy()
        p2 = im2[index].detach().cpu().permute(1,2,0).numpy()
        p1 = (p1 - p1.min()) / (p1.max() - p1.min())
        p2 = (p2 - p2.min()) / (p2.max() - p2.min())
        
        fig, ax = plt.subplots(1, 2)
        # plt.axis('off')
        ax1 = plt.subplot(1,2,1)
        ax1.axis('off')
        ax1.set_title('Model Input')
        ax1.imshow(p1)
        ax2 = plt.subplot(1,2,2)
        ax2.axis('off')
        ax2.set_title('Ground Truth')
        ax2.imshow(p2)
        fig.savefig(name)

    def drawOutOut(self, im1, im2, index, name):
        p1 = im1[index].detach().cpu().permute(1,2,0).numpy()
        p2 = im2[index].detach().cpu().permute(1,2,0).numpy()
        p1 = (p1 - p1.min()) / (p1.max() - p1.min())
        p2 = (p2 - p2.min()) / (p2.max() - p2.min())
        
        fig, ax = plt.subplots(1, 2)
        # plt.axis('off')
        ax1 = plt.subplot(1,2,1)
        ax1.axis('off')
        ax1.set_title('Model Output')
        ax1.imshow(p1)
        ax2 = plt.subplot(1,2,2)
        ax2.axis('off')
        ax2.set_title('Ground Truth')
        ax2.imshow(p2)
        fig.savefig(name)
