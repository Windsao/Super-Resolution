import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm

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
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            '''
            if epoch < self.args.start_aug:
                sr = self.model(lr, 0)
                loss = self.loss(sr, hr)
            else:
                sr = self.model(lr, 0)
                aug_img = self.attack_pgd(lr, hr, epsilon=self.args.eps, alpha=self.args.alpha, attack_iters=self.args.iters)
                aug_sr = self.model(aug_img, 0)
                loss = (1 - self.args.beta) * self.loss(sr,hr) + self.args.beta * self.loss(aug_sr,hr)
            '''
            t_sr = self.teacher(lr, 0)
            sr = self.model(lr, 0)
            loss = (1 - self.args.beta) * self.loss(sr,hr) + self.args.beta * self.loss(sr,t_sr)

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
        # cifar10_mean = (0.4914, 0.4822, 0.4465)
        # cifar10_std = (0.2471, 0.2435, 0.2616)

        # mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
        # std = torch.tensor(cifar10_std).view(3,1,1).cuda()

        # upper_limit = ((1 - mu)/ std)
        # lower_limit = ((0 - mu)/ std)
        # epsilon = epsilon / std
        # alpha = alpha / std
        # for zz in range(restarts):
        # for i in range(len(epsilon)):
        #     delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
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
    
    def _cutmix(im2, prob=1.0, alpha=1.0):
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

    def cutmix(im1, im2, prob=1.0, alpha=1.0):
        c = _cutmix(im2, prob, alpha)
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