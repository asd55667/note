#流程图
```
  digraph Inception1{
  	resolution=480;            
	dpi = 320;	

	a->b11->b12->b13->e
	a->b21->b22->e
	a->b31->b32->e
	a->b41->e

	a[shape=box,label="Input"];
	b11[shape=box,label="1x1"];
	b12[shape=box,label="3x3"];
	b13[shape=box,label="3x3"];

	b21[shape=box,label="1x1"];
	b22[shape=box,label="3x3"];

	b31[shape=box,label="Pooling"];
	b32[shape=box,label="1x1"];

	e[shape=box,label="Concatnate"];
	b41[shape=box,label="1x1"];
  }
```
```
digraph Inception2{
  	resolution=480;            
	dpi = 320;	
	a->b11->b12->b13->b14->b15->e
	a->b21->b22->b23->e
	a->b31->b32->e
	a->b41->e

	a[shape=box,label="Input"];

	b11[shape=box,label="1x1"];
	b12[shape=box,label="nx1"];
	b13[shape=box,label="1xn"];
	b14[shape=box,label="1xn"];
	b15[shape=box,label="nx1"];

	b21[shape=box,label="1x1"];
	b22[shape=box,label="1xn"];
	b23[shape=box,label="nx1"];

	b31[shape=box,label="Pooling"];
	b32[shape=box,label="1x1"];

	e[shape=box,label="Concatnate"];
	b41[shape=box,label="1x1"];
  }  
```
```
digraph inception3 {
  	resolution=480;            
	dpi = 320;	
	a->b11->b12->b131->e
	b12->b132->e

	a->b21->b221->e
	b21->b222->e

	a->b31->b32->e
	a->b41->e

	a[shape=box,label="Input"];

	b11[shape=box,label="1x1"];
	b12[shape=box,label="3x3"];
	b131[shape=box,label="1x3"];
	b132[shape=box,label="3x1"];

	b21[shape=box,label="1x1"];
	b221[shape=box,label="1x3"];
	b222[shape=box,label="3x1"];

	b31[shape=box,label="Pooling"];
	b32[shape=box,label="1x1"];

	e[shape=box,label="Concatnate"];
	b41[shape=box,label="1x1"];
  }    
  ```