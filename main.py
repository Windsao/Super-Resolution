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
            if args.distil:
                _teacher = model.Model(args, checkpoint)
                _teacher.load_state_dict(torch.load('/home/sw99/experiment/experiment/alpha_0.5_6/model/model_best.pt'))
                exit()
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
