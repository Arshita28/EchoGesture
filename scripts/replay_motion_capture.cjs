/* Replay reserved tracking sequences through the browser's automatic segmenter. */
const fs=require('node:fs');
const AutoMotion=require('../static/auto_motion.js');
const training=process.argv.includes('--training');
const expanded=process.argv.includes('--public-motion');
let input=JSON.parse(fs.readFileSync('datasets/preprocessed/motion-extension/'+(training?(expanded?'capture-public-training-input.json':'capture-training-input.json'):'sequences.json'),'utf8'));
const withStatic=!training&&process.argv.includes('--with-static');
if(withStatic)input=JSON.parse(fs.readFileSync('datasets/preprocessed/motion-extension/replay-with-static.json','utf8'));
if(!training&&!withStatic){
 const roles=JSON.parse(fs.readFileSync('datasets/preprocessed/motion-balanced-adaptation.json','utf8'));
 input=input.concat(roles.reserved_results.map(row=>({
  ...JSON.parse(fs.readFileSync('datasets/preprocessed/motion/'+row.file,'utf8')),
  path:row.file,role:'test',usable:true,source:'local_development'
 })));
}
(async()=>{
 const rows=[];
 for(const row of input.filter(r=>r.role===(training?'train':'test')&&r.usable)){
  const clips=[];
  const detector=new AutoMotion(async frames=>{
   clips.push(frames);
   return {status:'uncertain',candidate:'NONE',confidence:.5};
  });
  let lastTime=-Infinity;
  for(const frame of row.frames){
   if(frame.time-lastTime<95)continue;
   lastTime=frame.time;
   await detector.observe({...frame.static_result,landmarks:frame.points},frame.aspect,true,frame.time);
  }
  const last=row.frames.at(-1);
  for(let i=1;i<=12;i++)await detector.observe({...last.static_result,landmarks:last.points},last.aspect,true,last.time+i*100);
  rows.push({path:row.path,label:row.label,source_label:row.source_label,captured:clips.length,clips,phase:detector.phase,source:row.source||'dictionary_reserved'});
 }
 fs.writeFileSync('datasets/preprocessed/motion-extension/'+(training?(expanded?'capture-public-training.json':'capture-training.json'):'capture-replay.json'),JSON.stringify(rows));
 console.log(rows.map(r=>({label:r.source_label,captured:r.captured,phase:r.phase})));
})().catch(e=>{console.error(e);process.exitCode=1;});
