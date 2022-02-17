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
            temp = args.model
            if args.distil:
                args.model = 'EDSR'
                _teacher = model.Model(args, checkpoint)
                #_teacher.myload('../experiment/5_1_3/model/model_best.pt') 
                _teacher.myload('../experiment/baseline_edsr_x2_100_vaild/model/model_best.pt') 
            else:
                _teacher = None
            args.model = temp
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _model.myload('../experiment/baseline_edsr_x2_100_vaild/model/model_best.pt') 
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
