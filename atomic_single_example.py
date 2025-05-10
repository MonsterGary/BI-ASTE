import os
import sys
import argparse
import torch

sys.path.append(os.getcwd())

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_file", type=str, default='comet/pretrained_models/atomic_pretrained_model.pickle')
    parser.add_argument("--sampling_algorithm", type=str, default="help")

    args = parser.parse_args()
    model_file = 'comet/pretrained_models/atomic_pretrained_model.pickle'
    opt, state_dict = interactive.load_model_file(model_file)

    data_loader, text_encoder = interactive.load_data("atomic", opt)

    n_ctx = data_loader.max_event + data_loader.max_effect
    n_vocab = len(text_encoder.encoder) + n_ctx
    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

    if args.device != "cpu":
        cfg.device = int(args.device)
        cfg.do_gpu = True
        torch.cuda.set_device(cfg.device)
        model.cuda(cfg.device)
    else:
        cfg.device = "cpu"

    while True:
        input_event = "help"
        category = "help"
        sampling_algorithm = args.sampling_algorithm

        while input_event is None or input_event.lower() == "help":
            input_event = input("Give an event (e.g., PersonX went to the mall): ")

            if input_event == "help":
                interactive.print_help(opt.dataset)

        while category.lower() == "help":
            category = input("Give an effect type (type \"help\" for an explanation): ")

            if category == "help":
                interactive.print_category_help(opt.dataset)

        while sampling_algorithm.lower() == "help":
            sampling_algorithm = input("Give a sampling algorithm (type \"help\" for an explanation): ")

            if sampling_algorithm == "help":
                interactive.print_sampling_help()

        sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)

        if category not in data_loader.categories:
            category = "xReact"

        outputs = interactive.get_atomic_sequence(
            input_event, model, sampler, data_loader, text_encoder, category)
        print(outputs)

def initcomet():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_file", type=str, default='comet/pretrained_models/atomic_pretrained_model.pickle')
    parser.add_argument("--sampling_algorithm", type=str, default="help")

    args = parser.parse_args()
    model_file = 'comet/pretrained_models/atomic_pretrained_model.pickle'
    opt, state_dict = interactive.load_model_file(model_file)

    data_loader, text_encoder = interactive.load_data("atomic", opt)

    n_ctx = data_loader.max_event + data_loader.max_effect
    n_vocab = len(text_encoder.encoder) + n_ctx
    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

    if args.device != "cpu":
        cfg.device = int(args.device)
        cfg.do_gpu = True
        torch.cuda.set_device(cfg.device)
        model.cuda(cfg.device)
    else:
        cfg.device = "cpu"
    return args,opt,data_loader,model,text_encoder

def cometinference(args,opt,data_loader,model,text_encoder,input_event,category,sampling_algorithm):
    # input_event = "help"
    # category = "help"
    # sampling_algorithm = args.sampling_algorithm

    sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)

    if category not in data_loader.categories:
        category = "xReact"

    outputs = interactive.get_atomic_sequence(
        input_event, model, sampler, data_loader, text_encoder, category)
    # print(outputs)
    return outputs