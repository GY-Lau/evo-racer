/* One shared transform for the road, vehicles and sensor rays. */
(function(root){
 function create(){
  let mode='follow',zoom=8,center={x:700,y:420},scene;
  return {
   get mode(){return mode},get zoom(){return zoom},
   overview(){mode='overview';zoom=1;center={x:700,y:420}},
   follow(){mode='follow';zoom=8;center={x:700,y:420}},
   view(width,height,focus,key,preview=false){
    if(scene!==undefined&&key!==scene&&mode==='free')this.overview();
    scene=key;
    const z=preview&&mode==='follow'?1:zoom;
    const c=mode==='follow'&&!preview&&focus?{x:focus.x,y:focus.y}:center;
    const base=Math.min(width/1400,height/840),scale=base*z;
    return {width,height,base,scale,zoom:z,center:c,x:width/2-c.x*scale,y:height/2-c.y*scale};
   },
   wheel(view,x,y,delta){
    const z=Math.max(1,Math.min(32,view.zoom*Math.exp(-Math.max(-300,Math.min(300,delta))*.0015)));
    const anchor={x:view.center.x+(x-view.width/2)/view.scale,y:view.center.y+(y-view.height/2)/view.scale};
    zoom=z;mode='free';
    center={x:anchor.x-(x-view.width/2)/(view.base*z),y:anchor.y-(y-view.height/2)/(view.base*z)};
    if(z===1)this.overview();
   },
   pan(view,dx,dy){mode='free';zoom=view.zoom;center={x:view.center.x-dx/view.scale,y:view.center.y-dy/view.scale}}
  };
 }
 root.RaceCamera={create};
})(globalThis);
