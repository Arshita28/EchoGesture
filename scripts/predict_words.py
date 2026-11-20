"""Run a local isolated video through the word prototype, outputting JSON."""
import argparse
import json
from asl_lab.word_inference import predict

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('video');p.add_argument('--model',default=None)
    args=p.parse_args()
    try:print(json.dumps(predict(args.video,args.model),indent=2))
    except (ValueError,OSError) as exc:p.exit(2,str(exc)+'\n')
