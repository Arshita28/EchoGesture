/* Segment a complete gesture, classify once, and keep its honest outcome. */
class AutoMotion {
  constructor(send,notify=()=>{}) { this.send=send;this.notify=notify;this.epoch=0;this.reset(); }
  reset() {
    this.epoch++;this.frames=[];this.segment=[];this.phase='idle';this.busy=false;
    this.detected=null;this.anchor=null;this.lastPrediction=null;this.lastActive=-Infinity;this.finishedAt=-Infinity;
  }
  shape(frame) {
    const p=frame.points.map(v=>[v[0]-frame.points[0][0],(v[1]-frame.points[0][1])/frame.aspect]);
    const scale=Math.max(...p.map(v=>Math.hypot(...v)),1e-6);
    return p.map(v=>v.map(c=>c/scale));
  }
  shapeDifference(a,b) {return Math.sqrt(a.reduce((sum,p,i)=>sum+(p[0]-b[i][0])**2+(p[1]-b[i][1])**2,0)/21);}
  movement(frames=this.frames) {
    if(frames.length<2)return 0;
    const first=frames[0],p=first.points;
    const scale=Math.max(...p.map(v=>Math.hypot(v[0]-p[0][0],(v[1]-p[0][1])/first.aspect)));
    if(scale<1e-6)return 0;
    return Math.max(...[0,8,20].map(index=>{
      const xs=frames.map(f=>f.points[index][0]),ys=frames.map(f=>f.points[index][1]/f.aspect);
      return Math.hypot(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys))/scale;
    }));
  }
  current() {return this.phase==='recognized'?this.detected:null;}
  reading() {return this.phase!=='idle'&&this.phase!=='recognized';}
  feedback() {
    if(this.phase==='recording')return 'Reading motion — complete the gesture, then briefly hold still';
    if(this.phase==='classifying')return 'Checking the completed gesture…';
    if(this.phase==='incomplete')return 'Motion incomplete — change handshape or remove your hand, then retry';
    if(this.phase==='error')return 'Motion request failed — no static letter substituted';
    if(this.phase==='untrained')return 'Motion classifier is not trained';
    const p=this.lastPrediction;
    return p&&Number.isFinite(p.score)?`Motion uncertain · Closest ${p.candidate||'background'} · ${(p.score*100).toFixed(1)}% / 70% required. Change handshape or remove your hand to reset.`:'Motion uncertain — no static letter substituted';
  }
  incomplete(frame,now) {
    this.phase='incomplete';this.anchor=this.shape(frame);this.finishedAt=now;
    this.segment=[];this.notify(this.feedback());
  }
  async finish(frame,now) {
    const clip=this.segment.slice(),job=this.epoch;
    this.segment=[];this.phase='classifying';this.anchor=this.shape(frame);this.finishedAt=now;this.busy=true;
    this.notify(this.feedback());
    try {
      const p=await this.send(clip);if(job!==this.epoch)return;
      this.lastPrediction={candidate:p.candidate||p.label,score:p.confidence,at:now};
      if(p.status==='untrained')this.phase='untrained';
      else if(p.status==='ok'&&['J','Z'].includes(p.label)&&p.confidence>=.7) {
        this.phase='recognized';this.detected={label:p.label,score:p.confidence};
      }else this.phase='uncertain';
      this.notify(this.phase==='recognized'?`Completed motion: ${p.label}`:this.feedback());
    }catch(error){if(job===this.epoch){this.phase='error';this.notify(this.feedback());}}
    finally{if(job===this.epoch)this.busy=false;}
  }
  async observe(result,aspect,enabled,now=performance.now()) {
    if(!enabled||!result.landmarks){this.reset();return;}
    const previous=this.frames.at(-1);
    const frame={points:result.landmarks,time:now,aspect};
    if(previous&&now-previous.time>750) {
      if(this.phase==='recording'){this.incomplete(frame,now);this.frames=[frame];return;}
      this.reset();
    }
    this.frames.push(frame);this.frames=this.frames.filter(f=>now-f.time<=600).slice(-120);
    if(this.phase==='classifying')return;
    if(this.phase!=='idle'&&this.phase!=='recording'&&this.anchor&&this.shapeDifference(this.shape(frame),this.anchor)>.25) {
      this.phase='idle';this.detected=null;this.anchor=null;this.lastPrediction=null;
      this.finishedAt=now;this.frames=[frame];return;
    }
    if(this.phase==='recording') {
      this.segment.push(frame);
      if(previous&&this.movement([previous,frame])>.05)this.lastActive=now;
      const span=now-this.segment[0].time;
      // Wait for a real ending hold; never classify an ongoing timeout as complete.
      if(now-this.lastActive>=450&&this.segment.length>=8&&span>=700)await this.finish(frame,now);
      else if(span>4500||this.segment.length>=120)this.incomplete(frame,now);
      return;
    }
    const recent=this.frames.filter(f=>f.time>this.finishedAt);
    if(recent.length>=3&&this.movement(recent)>.3) {
      this.phase='recording';this.detected=null;this.lastPrediction=null;this.anchor=null;
      this.segment=recent.slice();this.lastActive=now;this.notify(this.feedback());
    }
  }
}
if(typeof window!=='undefined')window.AutoMotion=AutoMotion;
if(typeof module!=='undefined')module.exports=AutoMotion;
