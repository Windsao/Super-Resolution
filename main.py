import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            temp = {'model': args.model, 'n_resblocks': args.n_resblocks, 'n_feats':args.n_feats, 'res_scale': args.res_scale}
            if args.distil:
                args.model = 'EDSR'
                if args.teacher_model == 'EDSR_paper':
                    args.n_resblocks = 32
                    args.n_feats = 256 
                    args.res_scale = 0.1
                    _teacher = model.Model(args, checkpoint)
                    _teacher.myload('../SR_ckpt/EDSR_x4.pt')
                else:
                    _teacher = model.Model(args, checkpoint)
                    # _teacher.myload('../experiment/5_1_3/model/model_best.pt') 
                    _teacher.myload('../experiment/baseline2_EDSR_x4/model/model_best.pt')
                _teacher.eval() 
            else:
                _teacher = None
            args.model = temp['model']
            args.n_resblocks = temp['n_resblocks']
            args.n_feats = temp['n_feats']
            args.res_scale = temp['res_scale']
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint, _teacher)
            while not t.terminate():
                if args.distil:
                    t.distillation()
                else:
                    t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
