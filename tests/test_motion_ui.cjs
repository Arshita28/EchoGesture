const assert=require('node:assert/strict');
const mount=require('../static/motion.js');
let now=0,finish,posted=0;
const elements={};
global.document={getElementById(id){return elements[id] ||= {disabled:false,value:({motionAction:'collect',motionLabel:'J',motionGroup:'test',motionSplit:'validation'})[id]||'',textContent:''};}};
global.performance={now:()=>now};
global.setTimeout=fn=>{finish=fn;return 1;};global.clearTimeout=()=>{};
global.fetch=async(url,options)=>{posted++;const p=JSON.parse(options.body);return {ok:true,json:async()=>({status:'saved',frames:p.frames.length})};};
(async()=>{
 const ui=mount({ready:()=>true,speak:()=>{}});
 document.getElementById('motionRecord').onclick();
 for(let i=0;i<12;i++){now=i*180;ui.observe(i===4 ? {} : {landmarks:Array.from({length:21},()=>[.1,.2,.3])},1);}
 assert.equal(ui.active,true);await finish();assert.equal(posted,1);assert.match(elements.motionStatus.textContent,/Saved 11/);
 elements.motionRecord.onclick();
 for(let i=0;i<8;i++){now=i*150+(i>=4?1000:0);ui.observe({landmarks:Array.from({length:21},()=>[.1,.2,.3])},1);}
 await finish();assert.equal(posted,1);assert.match(elements.motionStatus.textContent,/lost for too long/);
 elements.motionRecord.onclick();ui.observe({},1);assert.equal(ui.active,true);ui.cancel();assert.equal(ui.active,false);
 console.log('Brief tracking gaps tolerated; incomplete clips rejected; Stop cancels correctly');
})().catch(error=>{console.error(error);process.exitCode=1;});
setImmediate(async()=>{
 try{
  let delay;global.setTimeout=(fn,ms)=>{finish=fn;delay=ms;return 1;};
  const ui=mount({ready:()=>true,speak:()=>{}});
  elements.motionRecord.onclick();assert.equal(delay,5000);
  ui.observe({},1);assert.equal(delay,5000);
  now=100;ui.observe({landmarks:Array.from({length:21},()=>[.1,.2,.3])},1);
  assert.equal(delay,2500);ui.cancel();
  elements.motionRecord.onclick();await finish();assert.equal(ui.active,false);
  assert.match(elements.motionStatus.textContent,/No hand detected/);
  console.log('Ready-hand timing and no-hand timeout checks passed');
 }catch(e){console.error(e);process.exitCode=1;}
});
setImmediate(async()=>{
 try{
  const requests=[];
  global.fetch=async(url,options)=>{
   const body=JSON.parse(options.body);requests.push(body);
   return {ok:true,json:async()=>body.action==='predict'?{status:'uncertain',candidate:'J',confidence:.55,threshold:.7}:{status:'saved',frames:body.frames.length}};
  };
  const ui=mount({ready:()=>true,speak:()=>{}});
  elements.motionAction.value='predict';elements.motionLabel.value='J';elements.motionRecord.onclick();
  for(let i=0;i<12;i++){now=i*180;ui.observe({landmarks:Array.from({length:21},()=>[.1,.2,.3])},1);}
  await finish();assert.equal(elements.motionReview.disabled,false);
  await elements.motionReview.onclick();
  assert.equal(requests[1].action,'collect');assert.equal(requests[1].split,'validation');
  assert.equal(requests[1].group,'motion_review');assert.equal(requests[1].label,'J');
  assert.deepEqual(requests[0].frames,requests[1].frames);
  assert.equal(elements.motionReview.disabled,true);
  console.log('Last recognition attempt saved exactly once for review, without another recording');
 }catch(e){console.error(e);process.exitCode=1;}
});
