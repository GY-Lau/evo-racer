/* Run the complete dashboard script against a GPU snapshot with a minimal DOM. */
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
module.exports=async function(config,snapshot,record){
 class Element{
  constructor(){this.children=[];this.dataset={};this.style={setProperty(){}};this.attributes={};this.nodes=new Map();this.classes=new Set();this.classList={toggle:(k,v)=>{if(v)this.classes.add(k);else this.classes.delete(k)},add:k=>this.classes.add(k),remove:k=>this.classes.delete(k)};this.clientWidth=800;this.clientHeight=300;this.hidden=false;this.textContent='';this._html=''}
  set innerHTML(v){this._html=v;const count=(v.match(/data-track=/g)||[]).length;if(count)this.children=Array.from({length:count},()=>new Element())}
  get innerHTML(){return this._html}
  setAttribute(k,v){this.attributes[k]=v}
  append(e){this.children.push(e)}
  insertBefore(e,b){this.children=this.children.filter(c=>c!==e);const i=this.children.indexOf(b);this.children.splice(i<0?this.children.length:i,0,e)}
  querySelector(k){if(!this.nodes.has(k))this.nodes.set(k,new Element());return this.nodes.get(k)}
  querySelectorAll(k){if(k==='.verification i'){if(!this.verify)this.verify=Array.from({length:5},()=>new Element());return this.verify}return []}
  getContext(){return new Proxy({},{get:(o,k)=>o[k]??(()=>{}),set:(o,k,v)=>(o[k]=v,true)})}
  addEventListener(){}scrollIntoView(){}showModal(){}close(){}
 }
 const nodes=new Map();const $=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id)};
 $('speeds').children=[1,3,10,30].map(n=>{const b=new Element();b.dataset.speed=String(n);return b});
 const errors=[];
 const box={document:{getElementById:$,createElement:()=>new Element(),documentElement:new Element(),querySelector:s=>s==='dialog[open]'?null:$(s),querySelectorAll:()=>[],addEventListener(){}},console:{warn:()=>{}},performance:{now:()=>1000},setTimeout:()=>0,requestAnimationFrame:()=>0,devicePixelRatio:1,ResizeObserver:class{observe(){}},matchMedia:()=>({matches:true}),AbortSignal:{timeout(){}},fetch:async url=>({ok:true,json:async()=>url==='/api/config'?config:url==='/api/records'?[record]:snapshot}),URL:{},Blob:class{}};
 vm.createContext(box);
 for(const f of ['motion.js','camera.js','app.js'])vm.runInContext(fs.readFileSync(path.join(__dirname,'../dist',f),'utf8'),box,{filename:f});
 // Flush setup, both API reads, archive loading, and the initial update.
 for(let i=0;i<20;i++)await Promise.resolve();
 assert.equal($('calendar').children.length,6);
 assert.equal($('standings').children.length,22);
 assert.match($('pedals').textContent,/油门/);
 assert.match($('cornering').textContent,/抓地力/);
 assert.equal($('past-champions').hidden,false);
 assert.match($('past-records').innerHTML,/BEARMAN|Bearman/);
 assert.match($('past-records').innerHTML,/165|02:45/);
 assert.equal($('pause').disabled,false);
 // The five-station standings must distinguish a skipped attempt from a pass.
 vm.runInContext("state.drivers[0].validationSkipped=[false,true,true,true,true];update()",box);
 assert.equal(vm.runInContext("rows.get(0).querySelectorAll('.verification i')[1].textContent",box),'—');
 assert.equal(vm.runInContext("rows.get(0).querySelectorAll('.verification i')[1].className",box),'skipped');
 console.log('PASS: full dashboard setup/update, physics telemetry, archive and skipped validation');
};
