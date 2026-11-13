/* Interpolate received physical states; never extrapolate or simulate driving. */
(function(root){
  function create({delay=300}={}){
    let frames=[];
    const scene=s=>`${s.generation}/${s.phase}/${s.track}/${s.holdout?.started??0}`;
    return {
      push(state,time){
        const last=frames.at(-1);
        if(!last||state.totalSteps<last.state.totalSteps||scene(last.state)!==scene(state)||state.paused||(state.winner&&state.phase!=='holdout')||last.state.paused){frames=[{state,time}];return}
        if(last.state.totalSteps===state.totalSteps)return;
        frames.push({state,time});
        if(frames.length>16)frames.shift();
      },
      sample(now){
        if(!frames.length)return [];
        const target=now-delay;
        while(frames.length>2&&frames[1].time<=target)frames.shift();
        const a=frames[0],b=frames[1];
        if(!b||target<=a.time)return a.state.cars;
        if(target>=b.time)return b.state.cars;
        const alpha=(target-a.time)/(b.time-a.time);
        const before=new Map(a.state.cars.map(c=>[c.id,c]));
        return b.state.cars.map(c=>{
          const p=before.get(c.id);
          if(!p||p.candidate!==c.candidate||Math.hypot(c.x-p.x,c.y-p.y)>180)return c;
          const da=((c.angle-p.angle+Math.PI)%(2*Math.PI)+2*Math.PI)%(2*Math.PI)-Math.PI;
          return {...c,x:p.x+(c.x-p.x)*alpha,y:p.y+(c.y-p.y)*alpha,angle:p.angle+da*alpha};
        });
      }
    };
  }
  root.RaceMotion={create};
})(globalThis);
