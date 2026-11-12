function mountMotion({ready,speak}) {
  const get=id=>document.getElementById(id);
  let frames=null,timer=null,saving=false,epoch=0,missed=0,waiting=false,lastAttempt=null;
  function controls(disabled) {
    get('motionReview').disabled=disabled||!lastAttempt;
    for(const id of ['motionRecord','motionAction','motionLabel','motionGroup','motionSplit'])get(id).disabled=disabled;
  }
  function cancel() {
    if(!frames&&!saving)return;
    epoch++;saving=false;waiting=false;frames=null;clearTimeout(timer);controls(false);
    get('motionStatus').textContent='Recording cancelled because the source stopped or a request failed.';
  }
  async function finish() {
    const job=epoch,recorded=frames;frames=null;waiting=false;saving=true;
    try {
      if(!recorded||recorded.length<8)throw Error(`Only ${recorded?recorded.length:0} usable frames captured. Clip not saved. Keep one hand well lit and inside the frame, then retry.`);
      if(recorded.at(-1).time-recorded[0].time<700)throw Error('Hand detected for too little of the recording. Retry with it already in view.');
      if(recorded.some((f,i)=>i>0&&f.time-recorded[i-1].time>750))throw Error('Hand tracking was lost for too long. Clip not saved; retry with smaller movements inside the frame.');
      const payload={action:get('motionAction').value,label:get('motionLabel').value,group:get('motionGroup').value,split:get('motionSplit').value,frames:recorded};
      const response=await fetch('/api/motion',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      const result=await response.json();if(job!==epoch)return;if(!response.ok)throw Error(result.error||'Motion request failed');
      if(payload.action==='predict')lastAttempt=recorded;
      if(result.status==='saved')get('motionStatus').textContent=`Saved ${result.frames} landmark frames locally. This does not train the model yet.`;
      else if(result.status==='untrained')get('motionStatus').textContent='Motion classifier is not trained yet.';
      else if(result.status==='ok'){
        get('motionStatus').textContent=`Experimental motion prediction: ${result.label} · Score ${(100*result.confidence).toFixed(1)}% · ${recorded.length} frames`;speak(result.label);
      }else{
        const name=result.candidate==='NONE'?'Background / no target motion':`Closest motion: ${result.candidate||'unknown'}`;
        get('motionStatus').textContent=`Uncertain · ${name} · Score ${(100*result.confidence).toFixed(1)}% / ${(100*(result.threshold||.7)).toFixed(0)}% required · ${recorded.length} frames`;
      }
    }catch(error){if(job===epoch)get('motionStatus').textContent=error.message;}
    finally{if(job===epoch){saving=false;controls(false);}}
  }
  get('motionReview').onclick=async()=>{
    if(!lastAttempt||saving||frames)return;
    saving=true;controls(true);
    try{
      const payload={action:'collect',label:get('motionLabel').value,group:'motion_review',split:'validation',frames:lastAttempt};
      const response=await fetch('/api/motion',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      const result=await response.json();if(!response.ok)throw Error(result.error||'Could not save review');
      lastAttempt=null;
      get('motionStatus').textContent=`Saved last attempt as ${payload.label} for local review only. It has not been added to training.`;
    }catch(error){get('motionStatus').textContent=error.message;}
    finally{saving=false;controls(false);}
  };
  get('motionRecord').onclick=()=>{
    if(!ready()||saving){get('motionStatus').textContent='Start the webcam and finish any other trial first.';return;}
    frames=[];missed=0;waiting=true;controls(true);
    get('motionStatus').textContent='Hold your starting hand pose in view. Recording starts when the hand is detected; then perform the motion.';
    timer=setTimeout(()=>{
      frames=null;waiting=false;controls(false);
      get('motionStatus').textContent='No hand detected within five seconds. Bring the whole hand into view and retry.';
    },5000);
  };
  return {get active(){return frames!==null||saving;},cancel,
    observe(result,aspect){
      if(!frames)return;
      if(!result.landmarks){
        missed++;
        get('motionStatus').textContent=waiting?'Waiting for one visible hand — hold the starting pose.':`Still recording · ${frames.length} usable frames · ${missed} missed. Bring one hand back into view.`;
        return;
      }
      if(waiting){waiting=false;missed=0;clearTimeout(timer);timer=setTimeout(finish,2500);}
      frames.push({points:result.landmarks,time:performance.now(),aspect});
      get('motionStatus').textContent=`Recording now — perform the motion once · ${frames.length} usable frames · ${missed} missed.`;
    }};
}
if(typeof window!=='undefined')window.mountMotion=mountMotion;
if(typeof module!=='undefined')module.exports=mountMotion;
