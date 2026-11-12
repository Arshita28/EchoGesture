const assert=require('node:assert/strict');const Auto=require('../static/auto_motion.js');
let clock=0;global.performance={now:()=>clock};
const base=Array.from({length:21},(_,i)=>[.2+i*.005,.3+i*.006,.01]);
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
 await gesture(uncertain);assert.equal(uncertain.current(),null);assert.equal(uncertain.phase,'uncertain');
 for(let i=15;i<50;i++)await feed(uncertain,i*100,.24);
 assert.equal(uncertain.reading(),true);assert.match(uncertain.feedback(),/uncertain/);
 // An unresolved ending pose cannot fall through to a static success.
 assert.equal(uncertain.current(),null);
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
