import torch

import utility
import data
import model
import loss
import os
from option import args
from trainer import Trainer

from fvcore.nn import FlopCountAnalysis, parameter_count_table

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
                    print('load large EDSR teacher!')
                else:
                    _teacher = model.Model(args, checkpoint)
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
            
            if args.resume_dir != '':
                path = os.path.join(args.resume_dir, 'model/model_best.pt')
                _model.myload(path)

            _loss = loss.Loss(args, checkpoint) if not args.test_only else None   
            t = Trainer(args, loader, _model, _loss, checkpoint, _teacher)

            # tensor = (torch.rand(1, 3, 64, 64).cuda(), 0)
            # flops = FlopCountAnalysis(_model, tensor)
            # print("Student FLOPs: ", flops.total())
            # print(parameter_count_table(_model))

            # t_flops = FlopCountAnalysis(_teacher, tensor)
            # print("Teacher FLOPs: ", t_flops.total())
            # print(parameter_count_table(_teacher))
            # exit()
    
            while not t.terminate():
                if args.distil:
                    t.distillation()
                else:
                    t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
