"""Replay frozen labeled clips through the local SVM with controlled video changes."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from urllib import request

import numpy as np

FILTERS = {
    'dim': 'lutrgb=r=val*0.55:g=val*0.55:b=val*0.55',
    'low_resolution': 'scale=160:-2',
    'occlusion': 'drawbox=x=iw*0.45:y=ih*0.35:w=iw*0.15:h=ih*0.20:color=black:t=fill',
    'holds': 'tpad=start_duration=1:stop_duration=1:start_mode=clone:stop_mode=clone',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    args = parser.parse_args()
    originals = json.loads(Path('reports/word20-svm-video-check.json').read_text())
    endpoint = 'http://127.0.0.1:8097/api/words'
    assert json.load(request.urlopen(endpoint+'/status'))['model'] == 'word20-citizen-svm'
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)/'variant.mp4'
        for condition, vf in FILTERS.items():
            for item in originals:
                source = next(Path('datasets/raw').rglob(item['clip']))
                subprocess.run([args.ffmpeg,'-loglevel','error','-y','-i',str(source),'-vf',vf,
                                '-an','-c:v','libx264','-crf','18',str(output)],check=True)
                reply = json.loads(subprocess.check_output(['curl','-sS','--max-time','40',
                    '-F','video=@'+str(output),endpoint+'/predict']))
                rows.append(dict(condition=condition,clip=item['clip'],expected=item['expected'],
                                 predicted=reply.get('label'),status=reply.get('status',reply.get('error')),
                                 ms=reply.get('processing_ms')))
            print(condition,sum(r['predicted']==r['expected'] for r in rows if r['condition']==condition),flush=True)
    Path('reports/word20-svm-robustness.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    main()
