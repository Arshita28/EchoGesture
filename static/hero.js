/* Original procedural animation, with an optional private local reference asset. */
(async()=>{
 const canvas=document.getElementById('heroArt');if(!canvas)return;
 const ctx=canvas.getContext('2d'),button=document.getElementById('heroPause');
 const reduced=matchMedia('(prefers-reduced-motion: reduce)');
 // The public fallback is generated independently; no reference coordinates/colors.
 function originalParticles() {
  const palette=[[128,82,255],[255,184,41],[255,255,255],[184,158,255],[21,132,110]];
  const n=6000,golden=Math.PI*(3-Math.sqrt(5));
  return Array.from({length:n},(_,i)=>{
   const y=1-2*(i+.5)/n,ring=Math.sqrt(1-y*y),theta=i*golden;
   const ripple=1+.055*Math.sin(theta*5+y*13)*Math.cos(y*17);
   const x=ring*Math.cos(theta),z=ring*Math.sin(theta);
   return [x*.94*ripple,y*.82*ripple,z*.68*ripple,...palette[i%palette.length]];
  });
 }
 let points=originalParticles();
 try {
  const r=await fetch('/static/brain-particles.json');
  if(r.ok) { const local=await r.json(); if(Array.isArray(local)&&local.length&&local.every(p=>Array.isArray(p)&&p.length===6&&p.every(Number.isFinite))) points=local; }
 } catch (_) { /* Original local canvas rendering requires no downloaded asset. */ }

 let w=0,h=0,raf=0,visible=true,paused=false,time=0,last=0;
 const ambient=Array.from({length:48},(_,i)=>({x:((i*137.51)%997)/997,y:((i*83.37)%991)/991,s:2+i%5,phase:i*1.9}));
 function triangle(x,y,size,angle,color,alpha){ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.lineWidth=.75;ctx.beginPath();for(let j=0;j<3;j++){const a=angle+j*Math.PI*2/3;const xx=x+Math.cos(a)*size,yy=y+Math.sin(a)*size;j?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy);}ctx.closePath();ctx.stroke();}
 function draw(){ctx.clearRect(0,0,w,h);const angle=-.14+Math.sin(time*.16)*.11,ca=Math.cos(angle),sa=Math.sin(angle),scale=Math.min(w*.49,h*.49);
  for(const a of ambient)triangle(a.x*w,a.y*h+Math.sin(time*.3+a.phase)*8,a.s,time*.08+a.phase,a.s%2?'#8052ff':'#ffb829',.18);
  for(let i=0;i<points.length;i++){const p=points[i],x=p[0]*ca+p[2]*sa,z=p[2]*ca-p[0]*sa,y=-p[1];const depth=(z+1)/2;const color=i%7===0?'#ffb829':`rgb(${p[3]},${p[4]},${p[5]})`;triangle(w*.5+x*scale,h*.51+y*scale,(1.2+depth*1.8)*Math.min(1,Math.max(.5,w/700)),time*.12+i*2.399,color,.25+depth*.7);}
  ctx.globalAlpha=1;
 }
 function resize(){const r=canvas.getBoundingClientRect();w=r.width;h=r.height;const d=Math.min(devicePixelRatio||1,2);canvas.width=w*d;canvas.height=h*d;ctx.setTransform(d,0,0,d,0,0);draw();}
 function tick(now){raf=0;if(last)time+=Math.min((now-last)/1000,.05);last=now;draw();if(visible&&!document.hidden&&!paused&&!reduced.matches)raf=requestAnimationFrame(tick);}
 function sync(){cancelAnimationFrame(raf);last=0;if(visible&&!document.hidden){draw();if(!paused&&!reduced.matches)raf=requestAnimationFrame(tick);}}
 button.onclick=()=>{paused=!paused;button.textContent=paused?'Play animation':'Pause animation';button.setAttribute('aria-pressed',String(paused));sync();};
 new ResizeObserver(resize).observe(canvas);new IntersectionObserver(([e])=>{visible=e.isIntersecting;sync();}).observe(canvas);document.addEventListener('visibilitychange',sync);reduced.addEventListener('change',sync);resize();sync();
})();
