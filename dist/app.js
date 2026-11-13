'use strict';
const $=id=>document.getElementById(id);
const motion=RaceMotion.create({delay:300});
const camera=RaceCamera.create();
let lastUI=0,dragging=null;
let config,state,selected=6,preview=null,sensors=true,online=false,winnerShown=false;
const backgrounds=new Map(), rows=new Map();
const canvas=$('track'),ctx=canvas.getContext('2d');
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const clock=(s,decimal=false)=>`${Math.floor(s/60).toString().padStart(2,'0')}:${(s%60).toFixed(decimal?1:0).padStart(decimal?4:2,'0')}`;
function toast(s){$('notice').textContent=s;$('notice').hidden=false;setTimeout(()=>$('notice').hidden=true,5000)}
async function control(action,value){try{const r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,value})});const v=await r.json();if(!r.ok)throw Error(v.error);return true}catch(e){toast(e.message);return false}}
function team(id){return config.teams.find(t=>t.id===config.drivers[id].team)}
function mini(t){return `<svg viewBox="0 0 1400 840" fill="none" stroke="currentColor" stroke-width="22"><path d="M${t.points.map(p=>p.join(',')).join('L')}Z"/></svg>`}
function setup(){
 $('roster-date').textContent=config.rosterDate;
 $('calendar').innerHTML=config.tracks.map((t,i)=>`<button class="circuit-card" data-track="${i}"><span class="card-number">${t.role==='holdout'?'UNSEEN TEST':'ROUND 0'+(i+1)} / ${t.country}</span><span class="station-status"></span><h3>${t.flag} ${t.short}</h3><p>${t.english}</p>${mini(t)}</button>`).join('');
 $('calendar').onclick=e=>{const b=e.target.closest('[data-track]');if(b){preview=Number(b.dataset.track);update();}};
 for(const d of config.drivers){const b=document.createElement('button');b.className='driver-row';b.style.setProperty('--driver',team(d.id).color);b.setAttribute('aria-label',d.name);b.innerHTML=`<span class="position"></span><span class="row-driver"><i class="team-stripe"></i><span><span class="row-name">${esc(d.code)} <small>${d.number??''}</small><span class="alive-marker"></span></span><span class="row-team">${esc(team(d.id).name)} · ${esc(d.zh)}</span></span></span><span class="verification">${config.tracks.filter(t=>t.role==='training').map(t=>`<i title="${t.short}">·</i>`).join('')}</span><span class="training-score"></span>`;b.onclick=()=>{if(state?.phase==='holdout'&&state.holdout.driver!==d.id)return;selected=d.id;camera.follow();syncCamera();update()};rows.set(d.id,b);$('standings').append(b)}
 $('roster-grid').innerHTML=config.teams.map(t=>`<div style="--driver:${t.color}"><strong>${esc(t.name)} / ${esc(t.zh)}</strong><p>${config.drivers.filter(d=>d.team===t.id).map(d=>esc(d.name)).join('<br>')}</p></div>`).join('');
 document.querySelector('.standings-note>p').textContent=`每位车手 ${config.candidates} 个独立训练模型，统一车模、相同规则。姓名与涂装用于识别实验个体。`;
 loadRecords();resize();
}
async function loadRecords(){
 try{const r=await fetch('/api/records',{cache:'no-store'});if(!r.ok)return;const records=await r.json();
 $('past-champions').hidden=!records.length;
 $('past-records').innerHTML=records.map(r=>{const w=r.winner,h=r.holdout,valid=/^[0-9]+$/.test(r.id);if(!valid)return '';const drivers=w.drivers.map(i=>config.drivers[i].name).join(' / ');return `<article><div><strong>${esc(drivers)}</strong><p>原物理版本冠军 · 第 ${w.generation} 轮 · 用时 ${clock(r.runtime)} · 五站各 3 圈通过</p><p>红牛环：${h.status==='passed'?'连续 3 圈通过':'未通过'} · ${clock(h.elapsed,true)} · 冻结模型 ${esc(h.modelHash.slice(0,12))}</p></div><div class="archive-links"><a href="/api/records/${r.id}/model" download="champion-${r.id}.json">↓ 冠军与完整赛事</a><a href="/api/records/${r.id}/holdout" download="holdout-${r.id}.json">↓ 红牛环记录</a></div></article>`}).join('');
 }catch(e){console.warn('历史成绩暂不可用',e)}
}
function path(c,points){c.beginPath();points.forEach((p,i)=>i?c.lineTo(...p):c.moveTo(...p));c.closePath()}
function background(index){
 if(backgrounds.has(index))return backgrounds.get(index);
 const t=config.tracks[index],p=t.points,s=t.metresScale,width=t.roadWidth,shape=new Path2D();
 p.forEach((v,i)=>i?shape.lineTo(...v):shape.moveTo(...v));shape.closePath();
 const draw=c=>{c.fillStyle='#14161b';c.fillRect(0,0,1400,840);c.lineJoin='round';c.lineCap='round';c.setLineDash([]);
 c.strokeStyle='#08090d';c.lineWidth=width+4*s;c.stroke(shape);
 c.strokeStyle='#c6cbd1';c.lineWidth=width+1.4*s;c.stroke(shape);
 c.strokeStyle='#dd3f39';c.setLineDash([3*s,3*s]);c.stroke(shape);c.setLineDash([]);
 c.strokeStyle='#30343d';c.lineWidth=width;c.stroke(shape);
 const [x,y]=p[0],a=Math.atan2(p[1][1]-y,p[1][0]-x);c.save();c.translate(x,y);c.rotate(a);
 for(let r=0;r<2;r++)for(let j=0;j<8;j++){c.fillStyle=(r+j)%2?'#fff':'#15171c';c.fillRect((r-1)*.6*s,j*width/8-width/2,.6*s,width/8)}c.restore();
 };backgrounds.set(index,draw);return draw;
}
function drawCar(c,car,id,viewScale){
 const t=team(id),focused=id===selected,s=state.trackScale,L=config.vehicle.length*s,W=config.vehicle.width*s;
 c.save();c.translate(car.x,car.y);c.rotate(car.angle);c.scale(L/26,W/16);c.globalAlpha=car.status===0?1:car.status===4?.7:.23;
 c.fillStyle='#07080b';const axle=config.vehicle.wheelbase/2/config.vehicle.length*26,tyreLength=.72/config.vehicle.length*26,tyreWidth=.32/config.vehicle.width*16;
 for(const x of [-axle,axle])for(const y of [-8,8-tyreWidth])c.fillRect(x-tyreLength/2,y,tyreLength,tyreWidth);
 c.fillStyle=t.color;c.beginPath();c.moveTo(-11,-4);c.lineTo(3,-4);c.lineTo(13,-1.5);c.lineTo(13,1.5);c.lineTo(3,4);c.lineTo(-11,4);c.closePath();c.fill();c.fillRect(10,-8,3,16);c.fillRect(-13,-7,3,14);c.fillStyle=car.brake>.1?'#ff2424':t.accent;c.fillRect(-8,-1,12,2);c.fillStyle='#090a0e';c.beginPath();c.ellipse(1,0,4,2.8,0,0,Math.PI*2);c.fill();c.restore();
 if(focused){const radius=Math.max(L*.8,7/viewScale);c.strokeStyle=t.color;c.lineWidth=1/viewScale;c.beginPath();c.arc(car.x,car.y,radius,0,Math.PI*2);c.stroke();c.font=`bold ${11/viewScale}px monospace`;c.textAlign='center';c.fillStyle=t.color;c.fillText(config.drivers[id].code,car.x,car.y-radius-6/viewScale)}
}
function syncCamera(){
 $('camera').textContent=camera.mode==='follow'?'切换全图':'跟车 8×';
 $('zoom-label').textContent=camera.zoom.toFixed(1)+'×';
}
function currentView(){return camera.view(canvas.clientWidth,canvas.clientHeight,motion.sample(performance.now())[selected],preview??state.track,(preview??state.track)!==state.track)}
canvas.addEventListener('wheel',e=>{
 if(!state||!config)return;e.preventDefault();
 const rect=canvas.getBoundingClientRect(),delta=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?rect.height:1);
 camera.wheel(currentView(),e.clientX-rect.left,e.clientY-rect.top,delta);syncCamera();
},{passive:false});
canvas.addEventListener('pointerdown',e=>{
 if(!state||e.button!==0)return;
 camera.pan(currentView(),0,0);dragging={id:e.pointerId,x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);canvas.classList.add('dragging');syncCamera();
});
canvas.addEventListener('pointermove',e=>{
 if(!dragging||dragging.id!==e.pointerId)return;
 camera.pan(currentView(),e.clientX-dragging.x,e.clientY-dragging.y);dragging.x=e.clientX;dragging.y=e.clientY;
});
function releaseDrag(e){if(dragging&&dragging.id===e.pointerId){dragging=null;canvas.classList.remove('dragging');if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId)}}
canvas.addEventListener('pointerup',releaseDrag);canvas.addEventListener('pointercancel',releaseDrag);
canvas.addEventListener('dblclick',()=>{camera.overview();syncCamera();update()});
for(const [id,delta] of [['zoom-in',-140],['zoom-out',140]])$(id).onclick=()=>{if(!state)return;const view=currentView();camera.wheel(view,view.width/2,view.height/2,delta);syncCamera();update()};
function frame(){
 if(config&&state){const w=canvas.clientWidth,h=canvas.clientHeight,dpr=Math.min(devicePixelRatio||1,2),base=Math.min(w/1400,h/840),track=preview??state.track;
 const cars=motion.sample(performance.now()),focus=cars[selected],view=camera.view(w,h,focus,track,track!==state.track),scale=view.scale;
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#14161b';ctx.fillRect(0,0,w,h);
 ctx.translate(view.x,view.y);ctx.scale(scale,scale);
 background(track)(ctx);
 const zoomText=view.zoom.toFixed(1)+'×';if($('zoom-label').textContent!==zoomText)$('zoom-label').textContent=zoomText;
 if(track===state.track){if(sensors&&focus?.status===0){[-1.4,-.9,-.45,0,.45,.9,1.4].forEach((a,i)=>{const angle=focus.angle+a,dist=focus.rays[i]*state.trackScale,x=focus.x+Math.cos(angle)*dist,y=focus.y+Math.sin(angle)*dist;ctx.strokeStyle=team(selected).color+'66';ctx.lineWidth=1/scale;ctx.beginPath();ctx.moveTo(focus.x,focus.y);ctx.lineTo(x,y);ctx.stroke();ctx.fillStyle=team(selected).color;ctx.fillRect(x-1/scale,y-1/scale,2/scale,2/scale)})}
 cars.filter(c=>!c.hiddenCar&&c.id!==selected).forEach(c=>drawCar(ctx,c,c.id,scale));if(focus&&!focus.hiddenCar)drawCar(ctx,focus,selected,scale)}
 const bar=50*config.tracks[track].metresScale*scale;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.strokeStyle='#d1d5de';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(22,h-42);ctx.lineTo(22+bar,h-42);ctx.stroke();ctx.fillStyle='#aeb6c5';ctx.font='10px monospace';ctx.textAlign='left';ctx.fillText('50 m',22,h-48);
 }requestAnimationFrame(frame);
}
function resize(){for(const id of ['track','brain','history']){const c=$(id),dpr=Math.min(devicePixelRatio||1,2);c.width=Math.max(1,Math.round(c.clientWidth*dpr));c.height=Math.max(1,Math.round(c.clientHeight*dpr))}if(state){brain();history()}}
function brain(){const c=$('brain'),p=c.getContext('2d'),w=c.clientWidth,h=c.clientHeight,dpr=Math.min(devicePixelRatio||1,2),car=state.cars[selected];p.setTransform(dpr,0,0,dpr,0,0);p.clearRect(0,0,w,h);if(!car)return;const counts=[8,10,2],values=[car.inputs,car.hidden,car.outputs],pos=counts.map((n,k)=>Array.from({length:n},(_,i)=>({x:9+(w-18)*k/2,y:8+(h-16)*(i+.5)/n})));let k=0;for(let l=0;l<2;l++)for(let j=0;j<counts[l+1];j++){for(let i=0;i<counts[l];i++){const weight=car.weights[k++],v=Math.min(1,Math.abs(weight*values[l][i]));p.strokeStyle=weight>0?`rgba(255,90,72,${.05+v*.5})`:`rgba(145,165,197,${.04+v*.3})`;p.lineWidth=.7;p.beginPath();p.moveTo(pos[l][i].x,pos[l][i].y);p.lineTo(pos[l+1][j].x,pos[l+1][j].y);p.stroke()}k++}pos.forEach((layer,l)=>layer.forEach((v,i)=>{p.fillStyle=values[l][i]>.2?team(selected).color:values[l][i]<-.2?'#a1acc0':'#414754';p.beginPath();p.arc(v.x,v.y,l===2?4.5:2.8,0,Math.PI*2);p.fill()}))}
function history(){const c=$('history'),p=c.getContext('2d'),w=c.clientWidth,h=c.clientHeight,dpr=Math.min(devicePixelRatio||1,2);p.setTransform(dpr,0,0,dpr,0,0);p.clearRect(0,0,w,h);p.font='9px monospace';for(let i=0;i<=4;i++){const y=10+(h-30)*i/4;p.fillStyle='#717a8d';p.fillText((100-i*25)+'%',0,y+3);p.strokeStyle='#353a46';p.setLineDash([2,4]);p.beginPath();p.moveTo(33,y);p.lineTo(w-5,y);p.stroke()}p.setLineDash([]);if(!state.history.length){p.fillStyle='#7e879a';p.textAlign='center';p.fillText('完成首轮五站训练后记录成绩',w/2,h/2);p.textAlign='left';return}for(const [key,color] of [['mean','#8490a7'],['minimum',team(selected).color]]){p.strokeStyle=color;p.lineWidth=1.7;p.beginPath();state.history.forEach((row,i)=>{const x=34+(w-42)*i/Math.max(12,state.history.length-1),y=10+(h-30)*(1-row[key][selected]);i?p.lineTo(x,y):p.moveTo(x,y)});p.stroke()}p.fillStyle='#717a8d';p.fillText('ROUND '+state.history[0].generation,34,h-3)}
function updateDriverStatus(row,car,driver){
 const alive=car?.status===0,complete=car?.status===4;
 row.classList.toggle('is-alive',alive);row.classList.toggle('is-complete',complete);
 row.querySelector('.alive-marker').textContent=alive?'● 存活':complete?'✓ 完成':'';
 const label=alive?'仍有模型存活':complete?'已完成当前赛段':'当前暂无存活模型';
 row.setAttribute('aria-label',driver.name+'，'+label);row.title=label;
}
function update(){if(!state||!config)return;const s=state,track=preview??s.track,t=config.tracks[track],d=config.drivers[selected],car=s.cars[selected],stats=s.drivers[selected],validation=s.phase!=='training',championshipDone=!!s.winner&&s.phase!=='holdout';document.documentElement.style.setProperty('--driver',team(selected).color);
 $('connection').textContent=online?'CUDA 训练器已连接':'连接中断，正在重试';$('connection-dot').classList.toggle('offline',!online);$('gpu').textContent=`${s.device.name} / CUDA ${s.device.cuda} · ${s.stepsPerSecond} 步/秒 · GPU 张量 ${s.gpuMemoryMB} MB`;$('network-count').textContent=`${s.networks} 个训练模型 · 22 位独立车手`;
 [...$('calendar').children].forEach((b,i)=>{b.classList.toggle('active',s.track===i);b.classList.toggle('preview',preview===i&&i!==s.track);b.querySelector('.station-status').textContent=s.track===i?(validation?'3 圈验证':'训练进行中'):(i===5?'未参与训练':'')});
 $('circuit-kicker').textContent=`ROUND 0${track+1} / ${t.country}`;$('circuit-name').textContent=t.name;$('phase').textContent=championshipDone?'CHAMPION':validation?'固定模型验证 · 3 圈':'训练评估 · 1 圈';$('phase').classList.toggle('validation',validation);$('follow-live').hidden=preview===null;$('view-label').textContent=track===s.track?(camera.mode==='follow'?'LIVE / FOLLOW':camera.mode==='free'?'LIVE / FREE VIEW':'LIVE / FULL CIRCUIT'):'CIRCUIT PREVIEW';$('view-description').textContent=track===s.track?'每位车手独立进化 · 直播展示当前代表车':`静态预览 · 实际训练在${config.tracks[s.track].short}站继续`;$('track-spec').textContent=`${(t.length/1000).toFixed(3)} KM / ${t.turns} TURNS / 路宽参考 ${t.widthMeters} m`;$('round-display').textContent=`TRAINING ROUND ${String(s.generation).padStart(3,'0')}`;
 $('validation-schedule').textContent=s.phase==='training'?`第 ${s.nextValidationGeneration} 轮结束后验证（每 ${s.validationInterval} 轮一次）`:'冻结模型 · 五站各连续 3 圈验证';$('stage-stat').textContent=championshipDone?'冠军已诞生':validation?'五站资格验证':'独立进化训练';$('stage-time').textContent=clock(s.elapsed,true);$('runtime').textContent=clock(s.runtime);$('active-models').innerHTML=`${s.active}<small> / ${validation?22:s.networks}</small>`;$('generation').textContent=String(s.generation).padStart(3,'0');$('mutation-label').textContent=Math.round(s.mutation*100)+'%';$('pause').textContent=s.paused?'▶ 继续赛事':'Ⅱ 暂停赛事';$('pause').disabled=!online||championshipDone;$('reset-open').disabled=!online;$('export').disabled=!online;for(const b of $('speeds').children){b.disabled=!online||championshipDone;b.classList.toggle('active',Number(b.dataset.speed)===s.speed)}
 $('race-overlay').hidden=online&&!s.paused&&!championshipDone;$('overlay-title').textContent=!online?'与 Jetson 的连接中断':championshipDone?'冠军已诞生':'赛事已暂停';$('overlay-detail').textContent=!online?'画面保留在最后收到的真实状态':championshipDone?config.drivers[s.winner.drivers[0]].name:'点击继续，从当前位置恢复';
 $('pedals').textContent=`油门 ${Math.round((car.throttle??0)*100)}% · 制动 ${Math.round((car.brake??0)*100)}%`;$('cornering').textContent=`横向 ${Math.abs(car.lateralG??0).toFixed(1)} g · 抓地力 ${Math.round((car.gripUsage??0)*100)}%`;$('selected-number').textContent=d.number??d.code;$('selected-name').textContent=d.name.split(' ').slice(-1)[0].toUpperCase();$('selected-team').textContent=team(selected).zh+' · '+d.zh;$('selected-country').textContent=d.country;$('selected-speed').innerHTML=`${Math.round(car.speed)}<small> km/h</small>`;$('selected-laps').textContent=car.laps.toFixed(2);$('selected-passes').innerHTML=`${stats.passed}<small> / 5</small>`;$('driver-status').textContent=`${car.reason} · ${validation?'冻结验证模型 '+stats.modelHash.slice(0,8):'当前训练代表模型 #'+(car.candidate%state.candidates+1)}`;
 s.ranking.forEach((id,rank)=>{const b=rows.get(id),r=s.drivers[id];updateDriverStatus(b,s.cars[id],config.drivers[id]);b.classList.toggle('selected',id===selected);b.setAttribute('aria-pressed',String(id===selected));b.querySelector('.position').textContent=String(rank+1).padStart(2,'0');b.querySelector('.training-score').innerHTML=`${Math.floor(r.minimum*100)}<small>%</small>`;b.querySelectorAll('.verification i').forEach((v,i)=>{const pass=r.validationTimes[i]!==null;const skipped=r.validationSkipped?.[i];v.className=pass?'passed':skipped?'skipped':validation&&s.track===i?'current':'';v.textContent=pass?'✓':skipped?'—':'·';v.title=config.tracks[i].short+(skipped?'：此前站验证失败，本站免跑':pass?`：3 圈通过，${r.validationTimes[i]} 秒`:`：未通过（本模型 ${r.validation[i].toFixed(2)} 圈）`)});if($('standings').children[rank]!==b)$('standings').insertBefore(b,$('standings').children[rank]||null)});
 $('best-passes').textContent=Math.max(...s.drivers.map(d=>d.passed))+' / 5';const events=JSON.stringify(s.events);if($('events').dataset.key!==events){$('events').dataset.key=events;$('events').innerHTML=s.events.map(e=>`<li class="${e.tag==='PASS'?'pass':''}"><span>R${String(e.generation).padStart(3,'0')} / ${esc(e.tag)}</span>${esc(e.text)}</li>`).join('')}
 $('finish-banner').hidden=!championshipDone;if(championshipDone){$('winner-name').textContent=s.winner.drivers.map(i=>config.drivers[i].name.toUpperCase()).join(' / ');$('winner-detail').textContent=`${s.winner.drivers.length>1?'并列冠军':'总冠军'} · 第 ${s.winner.generation} 轮 · 同一模型通过五站 · 验证累计 ${clock(s.winner.validationSeconds)}`;$('winner-circuits').innerHTML=config.tracks.filter(t=>t.role==='training').map(t=>`<span>✓ ${t.short} / 3 圈</span>`).join('');if(!winnerShown){winnerShown=true;$('finish-banner').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'})}}else winnerShown=false;
 updateHoldout();brain();history();
}
function updateHoldout(){
 const s=state,h=s.holdout||s.lastHoldout,testing=s.phase==='holdout';
 $('holdout-panel').hidden=!s.winner;
 $('holdout-start').disabled=!online||!s.winner;
 $('holdout-start').textContent=h?'↻ 重新测试冠军':'▶ 开始冠军测试';
 $('holdout-close').hidden=!testing;
 $('holdout-result').textContent=h?`${config.drivers[h.driver].code} · ${h.status==='running'?'试跑中':h.status==='passed'?'通过':'未通过'} · ${h.laps.toFixed(2)} / 3 圈 · ${clock(h.elapsed,true)} · ${h.reason}`:'原五站以外的全新赛道；冻结冠军模型，不学习、不变异，连续完成 3 圈算通过。';
 $('holdout-proof').textContent=h?`模型 ${h.modelHash.slice(0,12)} · ${h.weightsUnchanged?'权重校验一致':'权重校验异常'} · 未参与训练`:'测试结果独立保存，原赛事冠军与五站证书保留。';
 $('holdout-laps').textContent=h?h.lapSeconds.map((v,i)=>`第 ${i+1} 圈 ${v===null?'—':clock(v,true)}`).join('　'):'';
 for(const [id,b] of rows)b.disabled=testing&&id!==h.driver;
 if(!testing)return;
 $('connection').textContent=online?'CUDA 冠军泛化测试':'连接中断，正在重试';
 $('network-count').textContent='1 个冻结冠军模型 · 未知赛道测试';
 $('phase').textContent=h.status==='running'?'泛化测试 · 3 圈':h.status==='passed'?'泛化测试通过':'泛化测试未通过';
 if((preview??s.track)===5)$('circuit-kicker').textContent='UNSEEN TEST / AT';
 $('validation-schedule').textContent='红牛环从未参与训练与选模 · 原冠军权重冻结 · 目标连续 3 圈';
 $('stage-stat').textContent='冠军泛化测试';$('active-models').innerHTML=`${s.active}<small> / 1</small>`;
 $('round-display').textContent='FROZEN CHAMPION / UNSEEN CIRCUIT';
 $('view-description').textContent=(preview??s.track)===s.track?'仅冠军试跑 · 保持原物理规则与权重':'静态预览 · 冠军在红牛环测试';
 [...$('calendar').children].forEach((b,i)=>{b.querySelector('.station-status').textContent=i===s.track?'泛化测试':i===5?'未参与训练':''});
 $('driver-status').textContent=`${s.cars[selected].reason} · 冻结冠军模型 ${h.modelHash.slice(0,8)}`;
}
$('holdout-start').onclick=async()=>{if(await control('holdout',state.winner.drivers.includes(selected)?selected:null)){preview=null;camera.follow();syncCamera()}};
$('holdout-close').onclick=async()=>{if(await control('close_holdout')){preview=null;camera.follow();syncCamera()}};
async function poll(){const started=performance.now();try{if(!config){const r=await fetch('/api/config',{signal:AbortSignal.timeout(10000)});if(!r.ok)throw Error('赛历载入失败');config=await r.json();setup()}const r=await fetch('/api/state',{cache:'no-store',signal:AbortSignal.timeout(10000)});if(!r.ok)throw Error('训练服务响应异常');const next=await r.json();if(!next.ready)throw Error(next.error||'CUDA 训练器正在初始化');const changed=!state||state.phase!==next.phase||state.track!==next.track||state.generation!==next.generation||state.paused!==next.paused;if(next.phase==='holdout'&&(!state||state.phase!=='holdout'||state.holdout?.started!==next.holdout.started)){selected=next.holdout.driver;preview=null;camera.follow();syncCamera()}state=next;motion.push(next,performance.now());online=true;if(changed||performance.now()-lastUI>=250){update();lastUI=performance.now()}}catch(e){online=false;$('connection').textContent=e.message;$('connection-dot').classList.add('offline');if(state)update();for(const id of ['pause','export','reset-open'])$(id).disabled=true}setTimeout(poll,document.hidden?1000:Math.max(0,80-(performance.now()-started)))}
$('pause').onclick=()=>state&&control('pause',!state.paused);$('speeds').onclick=e=>{const b=e.target.closest('[data-speed]');if(b)control('speed',Number(b.dataset.speed))};$('camera').onclick=()=>{if(camera.mode==='follow')camera.overview();else camera.follow();syncCamera();update()};$('sensors').onclick=()=>{sensors=!sensors;$('sensors').classList.toggle('active',sensors);$('sensors').setAttribute('aria-pressed',String(sensors))};$('follow-live').onclick=()=>{preview=null;update()};
for(const [button,dialog] of [['rules-open','rules-dialog'],['reset-open','reset-dialog'],['roster-open','roster-dialog']])$(button).onclick=()=>$(dialog).showModal();
for(const button of document.querySelectorAll('[data-close]'))button.onclick=()=>$(button.dataset.close).close();
$('reset-confirm').onclick=async()=>{if(await control('reset')){$('reset-dialog').close();preview=null;camera.follow();syncCamera();winnerShown=false;toast('新赛季已开始，原赛事已归档。')}};
async function download(){if(!await control('save'))return;try{const r=await fetch('/api/model');if(!r.ok)throw Error('导出失败');const model=await r.json();const blob=new Blob([JSON.stringify(model)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`grid-evo-round-${state.generation}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}catch(e){toast(e.message)}}
$('export').onclick=download;$('winner-export').onclick=download;
document.addEventListener('keydown',e=>{if(e.code==='Space'&&!e.repeat&&!e.ctrlKey&&!e.metaKey&&!document.querySelector('dialog[open]')&&!/INPUT|BUTTON|SELECT|TEXTAREA/.test(e.target.tagName)&&online&&(!state?.winner||state.phase==='holdout')){e.preventDefault();control('pause',!state.paused)}});
new ResizeObserver(resize).observe(document.querySelector('.race-column'));poll();requestAnimationFrame(frame);
