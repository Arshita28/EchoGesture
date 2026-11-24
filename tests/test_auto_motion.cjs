const assert=require('node:assert/strict');const Auto=require('../static/auto_motion.js');
let clock=0;global.performance={now:()=>clock};
const base=Array.from({length:21},()=>[.2,.3,0]);
for(const mcp of [5,9,13,17]){
 base[mcp]=[.2+(mcp-11)*.005,.35,0];
 base[mcp+1]=[base[mcp][0],.4,0];
 base[mcp+2]=[base[mcp][0],.38,0];
 base[mcp+3]=[base[mcp][0],.36,0];
}
base[7]=[.17,.45,0];base[8]=[.17,.5,0];
function frame(move=0){return {landmarks:base.map(p=>[p[0]+move,p[1],p[2]])};}
async function feed(detector,time,move=0){clock=time;await detector.observe(frame(move),1,true,clock);}
async function gesture(detector){
 for(let i=0;i<=8;i++)await feed(detector,i*100,i*.03);
 for(let i=9;i<=14;i++)await feed(detector,i*100,.24);
}
(async()=>{
 let requests=[];const accepted=new Auto(async frames=>{requests.push(frames);return {status:'ok',label:'Z',candidate:'Z',confidence:.8};});
 for(let i=0;i<20;i++)await feed(accepted,i*100);
 assert.equal(requests.length,0);
 accepted.reset();await gesture(accepted);
 assert.equal(requests.length,1);assert.equal(accepted.current().label,'Z');assert.equal(accepted.reading(),false);
 assert.ok(requests[0].length>=8);assert.ok(requests[0].at(-1).time-requests[0][0].time<2000);
 // Holding an ending pose does not request overlapping stationary-tail windows.
 for(let i=15;i<=50;i++)await feed(accepted,i*100,.24);
 assert.equal(requests.length,1);assert.equal(accepted.current().label,'Z');
 const uncertain=new Auto(async()=>({status:'uncertain',candidate:'J',confidence:.69}));
 await gesture(uncertain);assert.equal(uncertain.current().label,'J');assert.equal(uncertain.current().accepted,false);assert.equal(uncertain.phase,'uncertain');
 for(let i=15;i<50;i++)await feed(uncertain,i*100,.24);
 assert.equal(uncertain.reading(),true);assert.match(uncertain.feedback(),/uncertain/);
 // An unresolved ending pose cannot fall through to a static success.
 assert.equal(uncertain.current().accepted,false);
 const changed=frame(.24);changed.landmarks[8][0]+=.5;clock=5100;
 await uncertain.observe(changed,1,true,clock);assert.equal(uncertain.phase,'idle');
 const background=new Auto(async()=>({status:'uncertain',candidate:'NONE',confidence:.95}));
 await gesture(background);assert.equal(background.current(),null);assert.equal(background.reading(),true);
 const short=new Auto(async()=>{throw Error('Should not classify too-short clip');});
 await feed(short,0);await feed(short,200,.04);await feed(short,400,.1);
 await feed(short,1400,.1);assert.equal(short.phase,'incomplete');
 let release;const pending=new Auto(()=>new Promise(r=>{release=r;}));
 for(let i=0;i<=8;i++)await feed(pending,i*100,i*.03);
 for(let i=9;i<=12;i++)await feed(pending,i*100,.24);
 clock=1300;const job=pending.observe(frame(.24),1,true,clock);assert.equal(pending.phase,'classifying');
 pending.reset();release({status:'ok',label:'J',confidence:.99});await job;
 assert.equal(pending.current(),null);assert.equal(pending.reading(),false);
 await accepted.observe({},1,true,5200);assert.equal(accepted.current(),null);
 await accepted.observe(frame(),1,false,5300);assert.equal(accepted.phase,'idle');
 console.log('Completed gesture classified once; pauses/uncertainty do not become static success; ending pose persists; late responses cancelled');
})().catch(error=>{console.error(error);process.exitCode=1;});

// An uncertain motion must release a steady static letter without hiding the hand.
(async()=>{
 const detector=new Auto(async()=>({status:'uncertain',candidate:'NONE',confidence:.9}));
 await gesture(detector);
 assert.equal(detector.phase,'uncertain');
 for(let i=15;i<=28;i++) {
  const result={...frame(.24),status:'ok',label:'A'};
  await detector.observe(result,1,true,i*100);
 }
 assert.equal(detector.phase,'idle');
 assert.equal(detector.reading(),false);
 // A completed J/Z yields to a different steady handshape.
 detector.phase='recognized';detector.detected={label:'Z',score:.9};detector.anchor=detector.shape({points:frame(.24).landmarks,aspect:1});
 const changed=frame(.24);changed.landmarks[8][0]+=.2;
 for(let i=30;i<=45;i++) await detector.observe({...changed,status:'ok',label:'B'},1,true,i*100);
 assert.equal(detector.phase,'idle');
 console.log('Steady static letters recover after uncertain motion and completed J/Z');
})().catch(error=>{console.error(error);process.exitCode=1;});

(async()=>{
 const q=Array.from({length:21},()=>[.5,.5,0]);
 q[8]=[.6,.8,0];q[18]=[.4,.65,0];q[20]=[.4,.6,0];
 const detector=new Auto(async()=>{throw Error('Steady Q should not start motion');});
 for(let i=0;i<20;i++){
  const points=q.map(p=>[p[0]+i*.01,p[1],p[2]]);
  await detector.observe({landmarks:points,status:'ok',label:'Q',confidence:.8},1,true,6000+i*100);
 }
 assert.equal(detector.phase,'idle');
 console.log('Confident downward Q is preserved during hand positioning');
})().catch(error=>{console.error(error);process.exitCode=1;});

(async()=>{
 const detector=new Auto(async()=>({status:'ok',label:'Z',confidence:.9}));
 for(let i=0;i<=8;i++) await detector.observe(frame(i*.03),1,true,10000+i*100);
 assert.equal(detector.phase,'recording');
 await detector.observe({},1,true,10900);
 assert.equal(detector.phase,'recording');
 for(let i=10;i<=16;i++) await detector.observe(frame(.24),1,true,10000+i*100);
 assert.equal(detector.current().label,'Z');
 console.log('Brief missing hand frame preserves the Z gesture');
})().catch(error=>{console.error(error);process.exitCode=1;});

(async()=>{
 const detector=new Auto(async()=>({status:'uncertain',candidate:'NONE',confidence:.99}));
 const little=base.map(p=>p.slice());
 little[8]=[.17,.36,0];little[19]=[.23,.45,0];little[20]=[.23,.5,0];
 for(let i=0;i<40;i++){
  const shift=Math.min(i,10)*.012;
  const points=little.map(p=>[p[0],p[1]-shift,p[2]]);
  await detector.observe({landmarks:points,status:'ok',label:'I',confidence:.95},1,true,20000+i*100);
  assert.notEqual(detector.current()?.label,'J');
 }
 assert.equal(detector.reading(),false);
 console.log('A straight I repositioning returns to static recognition without displaying J');
})().catch(error=>{console.error(error);process.exitCode=1;});
