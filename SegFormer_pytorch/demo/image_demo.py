from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette









def main():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file',type=str)
    parser.add_argument('--config', help='Config file',type=str)
    parser.add_argument('--checkpoint', help='Checkpoint file',type=str)
    parser.add_argument('--batchsize',help='Input batchsize',type=int)
    parser.add_argument('--size',type=int)
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    
   
    args = parser.parse_args()
    
    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_segmentor(model, args.img)
    # show the results
    show_result_pyplot(model, args.img, result, get_palette(args.palette))

if __name__ == '__main__':
    main()
