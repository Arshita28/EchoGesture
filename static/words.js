'use strict';
(() => {
 const el=id=>document.getElementById(id), camera=el('wordCamera');
 let stream=null,recorder=null,chunks=[],timer=null,busy=false,generation=0,controller=null,recordToken=0,starting=false;
 function controls(){el('wordStart').disabled=busy||starting||!!stream;el('wordRecord').disabled=busy||!stream;el('wordStop').disabled=!stream;el('wordUpload').disabled=busy;}
 function clearResult(){el('wordResult').textContent='—';el('wordDetails').textContent='';el('wordAlternatives').replaceChildren();}
 async function submit(blob,name){
  if(blob.size>8*1024*1024){busy=false;clearResult();controls();el('wordStatus').textContent='Video is too large. Use a clip under 8 MB.';return;}
  const token=++generation;busy=true;clearResult();controls();controller=new AbortController();
  el('wordStatus').textContent='Processing on this computer…';
  const form=new FormData();form.append('video',blob,name);
  try{
   const response=await fetch('/api/words/predict',{method:'POST',body:form,signal:controller.signal});
   const result=await response.json();if(token!==generation)return;
   if(!response.ok)throw Error(result.error||'Prediction failed');
   if(result.status==='insufficient_detection'){
    el('wordDetails').textContent='Not enough hand or body detection. Keep your face, shoulders and hands in view.';
   }else if(result.status==='uncertain_word'){
    el('wordDetails').textContent='No clear supported-word match. The sign may be unsupported or uncertain.';
   }else{
    el('wordResult').textContent=result.label;
    el('wordDetails').textContent='Experimental match. Hand detection '+Math.round(result.quality.any_hand_coverage*100)+'% of sampled frames.';
    for(const candidate of result.candidates){const li=document.createElement('li');li.textContent=candidate.label;el('wordAlternatives').append(li);}
   }
   el('wordStatus').textContent='Finished in '+(result.processing_ms/1000).toFixed(1)+' seconds. Try another isolated sign.';
  }catch(error){if(token===generation&&error.name!=='AbortError'){el('wordStatus').textContent=error.message;el('wordDetails').textContent='No result. Try a short, supported video.';}}
  finally{if(token===generation){busy=false;controller=null;controls();}}
 }
 el('wordStart').onclick=async()=>{
  const token=++generation;starting=true;controls();
  try{
   if(!navigator.mediaDevices?.getUserMedia)throw Error('Camera access is unavailable. Upload a video instead.');
   const incoming=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480}},audio:false});
   if(token!==generation){incoming.getTracks().forEach(t=>t.stop());return;}
   stream=incoming;camera.srcObject=stream;await camera.play();el('cameraHint').hidden=true;
   el('wordStatus').textContent='Camera preview ready. Click Record 3 seconds to run recognition. Sign once with your face, shoulders and hands visible.';
  }catch(error){if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}el('wordStatus').textContent=error.message;}
  finally{starting=false;controls();}
 };
 el('wordRecord').onclick=()=>{
  if(!stream||busy)return;
  if(!window.MediaRecorder){el('wordStatus').textContent='Recording is unavailable in this browser. Upload a video instead.';return;}
  clearResult();const token=++recordToken;chunks=[];busy=true;controls();
  const type=['video/webm;codecs=vp8','video/webm','video/mp4'].find(t=>MediaRecorder.isTypeSupported(t));
  if(!type){busy=false;controls();el('wordStatus').textContent='No supported recording format. Upload a video instead.';return;}
  try{
   recorder=new MediaRecorder(stream,{mimeType:type,videoBitsPerSecond:1200000});
   recorder.ondataavailable=event=>{if(event.data.size)chunks.push(event.data);};
   recorder.onerror=()=>{clearTimeout(timer);recordToken++;busy=false;controls();el('wordStatus').textContent='Recording failed. Try uploading a video.';};
   recorder.onstop=()=>{clearTimeout(timer);if(token!==recordToken)return;const blob=new Blob(chunks,{type});submit(blob,type.includes('mp4')?'sign.mp4':'sign.webm');};
   recorder.start();let remaining=3;el('wordStatus').textContent='Recording: 3 seconds. Sign once now.';
   timer=setInterval(()=>{remaining--;if(remaining>0)el('wordStatus').textContent='Recording: '+remaining+' seconds…';else{clearInterval(timer);if(recorder.state==='recording')recorder.stop();}},1000);
  }catch(error){busy=false;controls();el('wordStatus').textContent=error.message;}
 };
 function stop(){generation++;recordToken++;clearInterval(timer);controller?.abort();controller=null;
  if(recorder&&recorder.state!=='inactive')recorder.stop();recorder=null;
  if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;camera.srcObject=null;
  busy=false;el('cameraHint').hidden=false;controls();
 }
 el('wordStop').onclick=()=>{stop();clearResult();el('wordStatus').textContent='Stopped. No pending result will be displayed.';};
 el('wordUpload').onchange=event=>{const file=event.target.files[0];if(file&&!busy)submit(file,file.name);event.target.value='';};
 el('wordExample').onchange=event=>{el('exampleVideo').src='/api/words/example/'+encodeURIComponent(event.target.value);el('exampleVideo').load();};
 addEventListener('pagehide',stop);controls();
})();
