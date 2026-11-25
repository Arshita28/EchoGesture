const $ = id => document.getElementById(id);
let stream = null, objectURL = null, generation = 0, running = false;
let history = [], current = null, lastCapture = 0;
let activeImageFile = null;
let modelReady = false;
// Use completed motion gestures alongside static alphabet recognition.
const LIVE_CUTOFF = 30;
const AUTO_MOTION_ENABLED = true;
let modelName = "Unavailable";
const letterVoice = new LetterVoice(window.speechSynthesis, window.SpeechSynthesisUtterance, text => { $('voiceStatus').textContent = text; });
if (!letterVoice.supported) { $('voiceToggle').disabled = true; $('voiceStatus').textContent = 'Voice output is unavailable in this browser'; }
$('voiceToggle').onclick = () => {
  letterVoice.enable(!letterVoice.enabled);
  $('voiceToggle').textContent = letterVoice.enabled ? 'Disable voice' : 'Enable voice';
  $('voiceToggle').setAttribute('aria-pressed', String(letterVoice.enabled));
  if (current) letterVoice.observe(current, 'ok');
};
const evaluation = mountEvaluation({ready:()=>Boolean(stream && running && modelReady && !motion.active), cutoff:()=>LIVE_CUTOFF, model:()=>modelName});
const motion = mountMotion({ready:()=>Boolean(stream && running && !evaluation.active),speak:letter=>letterVoice.observe(letter,'ok')});
const autoMotion = new AutoMotion(async frames => {
  const response=await fetch('/api/motion',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'predict',frames})});
  const result=await response.json();if(!response.ok)throw Error(result.error||'Motion request failed');return result;
});
const messages = {
  ok: 'Letter recognized', uncertain: 'Uncertain — adjust your hand',
  no_hand: 'No hand detected', multiple_hands: 'Show one hand only',
  untrained: 'No trained classifier yet'
};
const edges = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
  [13,17],[0,17],[17,18],[18,19],[19,20]];

function clearOverlay() {
  const canvas = $('overlay');
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}
function reset(clearSpeech = true) {
  history = []; current = null; if(clearSpeech) letterVoice.observe(null, 'cleared'); $('letter').textContent = '—'; clearOverlay();
}
function stop() {
  generation++; running = false; autoMotion.reset(); letterVoice.reset(); evaluation.cancel(); motion.cancel();
  activeImageFile = null;
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  $('video').pause(); $('video').srcObject = null;
  $('video').removeAttribute('src'); $('video').classList.remove('active');
  $('video').load();
  if (objectURL) URL.revokeObjectURL(objectURL);
  objectURL = null;
  $('photo').hidden = true; $('empty').hidden = false;
  $('source').textContent = 'NO SOURCE'; reset();
}
async function frame() {
  const media = $('photo').hidden ? $('video') : $('photo');
  const width = media.videoWidth || media.naturalWidth;
  const height = media.videoHeight || media.naturalHeight;
  if (!width || !height) throw Error('No decoded media frame available');
  const scale = Math.min(1, 960 / width), canvas = $('canvas');
  canvas.width = Math.round(width * scale); canvas.height = Math.round(height * scale);
  canvas.getContext('2d').drawImage(media, 0, 0, canvas.width, canvas.height);
  return new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
}
function drawLandmarks(points) {
  clearOverlay();
  if (!points) return;
  const canvas = $('overlay');
  canvas.width = $('canvas').width; canvas.height = $('canvas').height;
  const ctx = canvas.getContext('2d');
  ctx.strokeStyle = '#c4f078'; ctx.fillStyle = '#c4f078'; ctx.lineWidth = 2;
  for (const [a,b] of edges) {
    ctx.beginPath(); ctx.moveTo(points[a][0]*canvas.width, points[a][1]*canvas.height);
    ctx.lineTo(points[b][0]*canvas.width, points[b][1]*canvas.height); ctx.stroke();
  }
  for (const [x,y] of points) {
    ctx.beginPath(); ctx.arc(x*canvas.width, y*canvas.height, 3, 0, 2*Math.PI); ctx.fill();
  }
}
async function post(url, data) {
  const response = await fetch(url, {method: 'POST', body: data});
  const result = await response.json();
  if (!response.ok) throw Error(result.error || 'Request failed');
  return result;
}
async function predict(token, staticImage = false, originalFile = null) {
  if (token !== generation) return;
  try {
    const blob = originalFile || await frame();
    if (token !== generation) return;
    const data = new FormData(); data.append('image', blob, originalFile ? originalFile.name : 'frame.png');
    data.append('cutoff', LIVE_CUTOFF);
    const result = await post('/api/predict', data);
    if (token !== generation) return;
    // A control change while a request is in flight must use the latest cutoff.
    if (result.confidence != null) {
      result.threshold = Number(LIVE_CUTOFF) / 100;
      result.status = result.confidence >= result.threshold ? 'ok' : 'uncertain';
      result.label = result.status === 'ok' ? result.candidate : null;
    }
    if (result.status === 'ok') {
      history.push(result.label); history = history.slice(-4);
      current = staticImage || (history.length === 4 && history.every(x => x === result.label))
        ? result.label : null;
    } else reset(false);
    const useAuto = Boolean(AUTO_MOTION_ENABLED && !staticImage && running && !motion.active && !evaluation.active);
    autoMotion.observe(result, $('canvas').width / $('canvas').height, useAuto);
    const motionLetter=useAuto ? autoMotion.current() : null;
    const readingMotion=useAuto && !motionLetter && autoMotion.reading();
    if(readingMotion) current=null;
    if(motionLetter) current=motionLetter.label;
    drawLandmarks(result.landmarks);
    $('letter').textContent = current || '—';
    letterVoice.observe(current, motionLetter ? (motionLetter.accepted ? 'ok' : 'uncertain') : result.status);
    evaluation.observe(current, result);
    if (!staticImage) motion.observe(result, $('canvas').width / $('canvas').height);
    $('state').textContent = result.status === 'ok' && !current
      ? 'Hold steady — confirming sign' : messages[result.status] || result.status;
    if (result.confidence != null) {
      const candidate = result.status === 'uncertain' ? `Closest candidate ${result.candidate} · ` : '';
      $('score').textContent = `${candidate}Score ${(result.confidence*100).toFixed(1)}% / ${(result.threshold*100).toFixed(0)}% required · ${result.processing_ms} ms server`;
    } else {
      const hints = {
        no_hand: 'Keep your whole hand visible, well lit and away from your face.',
        multiple_hands: 'Remove the second hand from the frame.',
        untrained: 'Collect samples, extract landmarks and train the classifier.'
      };
      $('score').textContent = hints[result.status] || 'Choose a source to begin.';
    }
    if(motionLetter) {
      $('state').textContent=motionLetter.accepted?'Motion letter recognized':'Best motion prediction — uncertain';
      $('score').textContent=motionLetter.method==='trajectory'
        ? `${motionLetter.label} · Finger shape and completed path matched`
        : `${motionLetter.label} · Model score ${(motionLetter.score*100).toFixed(1)}%`;
    }
    else if(readingMotion) {
      $('state').textContent=autoMotion.phase==='recording'?'Reading motion — static letters paused':autoMotion.phase==='classifying'?'Checking completed motion':'Motion not confirmed';
      $('score').textContent=autoMotion.feedback();
    } else if(useAuto && current) $('state').textContent='Static letter recognized';
  } catch (error) {
    if (token === generation) { evaluation.cancel(); motion.cancel(); autoMotion.reset(); reset(); $('state').textContent = error.message; }
  }
  if (token === generation && running && !staticImage) {
    if ($('video').ended) {
      running = false; letterVoice.reset(); reset(); $('state').textContent = 'Video finished'; return;
    }
    setTimeout(() => predict(token), (motion.active || (AUTO_MOTION_ENABLED && running)) ? 40 : 180);
  }
}
$('camera').onclick = async () => {
  stop(); const token = generation;
  try {
    const acquired = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
    if (token !== generation) { acquired.getTracks().forEach(t => t.stop()); return; }
    stream = acquired; $('video').srcObject = stream; $('video').classList.add('active');
    await $('video').play(); $('empty').hidden = true;
    $('source').textContent = 'LIVE CAMERA'; running = true; predict(token);
  } catch (error) { $('state').textContent = `Camera: ${error.message}`; }
};
$('stop').onclick = () => { stop(); $('state').textContent = 'Stopped'; };
$('upload').onchange = async event => {
  const file = event.target.files[0];
  if (!file) return;
  stop(); const token = generation;
  objectURL = URL.createObjectURL(file); $('empty').hidden = true;
  $('source').textContent = file.name;
  const isImage = file.type.startsWith('image/') || /\.(png|jpe?g|webp|gif|bmp|avif)$/i.test(file.name);
  const isVideo = file.type.startsWith('video/') || /\.(mp4|mov|m4v|webm|ogv)$/i.test(file.name);
  if (isImage) {
    activeImageFile = file;
    $('photo').onload = () => {
      $('canvas').width = $('photo').naturalWidth;
      $('canvas').height = $('photo').naturalHeight;
      predict(token, true, file);
    };
    $('photo').onerror = () => { $('state').textContent = 'Could not decode image'; };
    $('photo').src = objectURL; $('photo').hidden = false;
  } else if (isVideo) {
    try {
      $('video').src = objectURL; $('video').classList.add('active');
      await $('video').play(); running = true; predict(token);
    } catch (error) {
      $('state').textContent = 'This browser cannot play this video format';
      $('score').textContent = 'Try an MP4 with H.264 video. Photo Booth MOV files may need conversion. Try uploading the converted video again.';
    }
  } else $('state').textContent = 'Choose an image or video';
  event.target.value = '';
};
function editMessage(value) { letterVoice.reset(); $('text').value = value; }
$('append').onclick = () => { if (current) editMessage($('text').value + current); };
$('space').onclick = () => { const value=$('text').value; if(value && !value.endsWith(' ')) editMessage(value+' '); };
$('backspace').onclick = () => editMessage(Array.from($('text').value).slice(0,-1).join(''));
$('speakMessage').onclick = () => letterVoice.speakMessage($('text').value);
$('stopSpeech').onclick = () => { letterVoice.reset(); $('voiceStatus').textContent='Speech stopped'; };
$('copyMessage').onclick = async () => {
  try { await navigator.clipboard.writeText($('text').value); $('messageStatus').textContent='Message copied'; }
  catch (_) { $('text').focus(); $('text').select(); $('messageStatus').textContent='Text selected — use Copy on your device'; }
};
$('text').addEventListener('input',()=>letterVoice.reset());
$('clear').onclick = () => editMessage('');
$('capture').onclick = async () => {
  if (Date.now() - lastCapture < 1000) return;
  lastCapture = Date.now(); $('capture').disabled = true;
  try {
    const data = new FormData(); data.append('image', await frame(), 'capture.jpg');
    for (const id of ['group','split','label']) data.append(id, $(id).value);
    const result = await post('/api/collect', data);
    $('notice').textContent = `Saved. ${result.samples} total collected samples.`;
  } catch (error) { $('notice').textContent = error.message; }
  finally { $('capture').disabled = false; }
};
fetch('/api/status').then(r => r.json()).then(status => {
  modelReady = status.trained;
  modelName = status.model || 'Unavailable';
  $('model').textContent = status.trained ? 'Ready to recognize' : 'Recognition unavailable';
}).catch(() => { $('model').textContent = 'Server unavailable'; });
window.addEventListener('beforeunload', stop);
