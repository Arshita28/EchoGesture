const assert=require('node:assert/strict');const fs=require('node:fs');const vm=require('node:vm');
const elements={};const el=id=>elements[id]??={disabled:false,textContent:'',value:'',files:[],hidden:false,replaceChildren(){},append(){},play:async()=>{},load(){}};
let stopCount=0,timer,fetchCount=0,resolveFetch;
const stream={getTracks:()=>[{stop(){stopCount++;}}]};
class Recorder{static isTypeSupported(t){return t.includes('webm');}constructor(){this.state='inactive';}start(){this.state='recording';}stop(){this.state='inactive';this.ondataavailable({data:new Blob(['fixture'])});this.onstop();}}
const response=()=>({ok:true,json:async()=>({status:'prototype_match',label:'computer',quality:{any_hand_coverage:1},candidates:[{label:'computer'}],processing_ms:1000})});
const context={document:{getElementById:el,createElement:()=>({})},navigator:{mediaDevices:{getUserMedia:async()=>stream}},
 MediaRecorder:Recorder,window:{MediaRecorder:Recorder},Blob,FormData:class{append(){}},AbortController,
 setInterval:fn=>{timer=fn;return 1;},clearInterval(){},clearTimeout(){},addEventListener(){},
 fetch:async()=>{fetchCount++;return response();},console};
vm.runInNewContext(fs.readFileSync('static/words.js','utf8'),context);
const flush=()=>new Promise(r=>setImmediate(r));
(async()=>{
 assert.equal(el('wordRecord').disabled,true);await el('wordStart').onclick();assert.equal(el('wordRecord').disabled,false);
 el('wordRecord').onclick();timer();timer();timer();await flush();
 assert.equal(fetchCount,1);assert.equal(el('wordResult').textContent,'computer');assert.equal(el('wordRecord').disabled,false);
 el('wordUpload').onchange({target:{files:[{name:'large.webm',size:9*1024*1024}],value:'x'}});await flush();
 assert.match(el('wordStatus').textContent,/too large/);assert.equal(el('wordRecord').disabled,false);
 context.fetch=async()=>({ok:true,json:async()=>({status:'uncertain_word',label:null,processing_ms:1000})});
 el('wordUpload').onchange({target:{files:[{name:'uncertain.webm',size:10}],value:'x'}});await flush();
 assert.equal(el('wordResult').textContent,'—');assert.match(el('wordDetails').textContent,/No clear supported-word/);
 context.fetch=()=>{fetchCount++;return new Promise(r=>{resolveFetch=r;});};
 el('wordUpload').onchange({target:{files:[{name:'small.webm',size:10}],value:'x'}});
 el('wordStop').onclick();resolveFetch(response());await flush();
 assert.equal(el('wordResult').textContent,'—');assert.match(el('wordStatus').textContent,/Stopped/);assert.ok(stopCount>0);
 console.log('Word recording submits once; oversize resets controls; Stop releases camera and discards stale results');
})().catch(e=>{console.error(e);process.exitCode=1;});
