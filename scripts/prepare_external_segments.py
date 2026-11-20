"""Trim web evaluation sources using label-specific official 25-FPS annotations."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ffmpeg',required=True);a=p.parse_args()
    folder=Path('datasets/raw/web-evaluation');sources=json.loads((folder/'sources.json').read_text())
    metadata=json.loads(Path('datasets/raw/wlasl/WLASL_v0.3.json').read_text())
    for row in sources:
        if row['status']!='downloaded' or row['source']!='wlasl_unacquired':continue
        matches=[i for e in metadata if e['gloss']==row['label'] for i in e['instances'] if i['url']==row['url']]
        if len(matches)!=1:raise ValueError('Expected one label-specific source annotation')
        item=matches[0];start=(item['frame_start']-1)/item['fps'];duration=(item['frame_end']-item['frame_start']+1)/item['fps']
        raw=Path(row.get('raw_path',row['path']));output=folder/(row['label']+'-'+row['video_id']+'-segment.mp4')
        subprocess.run([a.ffmpeg,'-v','error','-y','-i',str(raw),'-ss',str(start),'-t',str(duration),'-an','-c:v','libx264','-crf','18',str(output)],check=True)
        row.update(raw_path=str(raw),path=str(output.resolve()),segment=dict(start_s=start,duration_s=duration,official_frame_start=item['frame_start'],official_frame_end=item['frame_end'],annotation_fps=item['fps'],wlasl_video_id=item['video_id']))
        print(row['label'],start,duration)
    (folder/'sources-segmented.json').write_text(json.dumps(sources,indent=2)+'\n')

if __name__=='__main__':main()
