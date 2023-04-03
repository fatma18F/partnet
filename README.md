## Visualization of Pointnet part segmentation 

model trained on partnet dataset: category chair level 1 

Part Name List:  [
- chair/chair_head,
- chair/chair_back, 
- chair/chair_arm, 
- chair/chair_base,
- chair/chair_seat ]

#### part segmentation result:

Exp1             | Exp2
:-------------------------:|:-------------------------:
![Screenshot](./photos/newplot43.png) |  ![Screenshot](./newplot5.png)


#### part instance segmentation result:

Exp1             | Exp2
:-------------------------:|:-------------------------:
![Screenshot](./photos/newplot41.png) |  ![Screenshot](./photos/newplot42.png) 

### per instance visualisation:
1. 
```
n_instances=0
for i in range(0,200):
   gt_instance=batch['masks'][1][i].float()
   n=np.unique(gt_instance.ravel()).size
   if n>1:
     n_instances+=1
n_instances
```
total instaces are 3 


instance1             | instance2             | instance3 
:-------------------------:|:-------------------------:|:-------------------------:
![Screenshot](./photos/newplot52.png) |  ![Screenshot](./photos/newplot53.png)  |  ![Screenshot](./photos/newplot51.png) 

=>the 4 legs are in the same instace

2.

total instaces are 5

instance1             | instance2             | instance3 
:-------------------------:|:-------------------------:|:-------------------------:
![Screenshot](./photos/newplot61.png) |  ![Screenshot](./photos/newplot62.png)  |  ![Screenshot](./photos/newplot63.png) 
instance4             | instance5             
:-------------------------:|:-------------------------:
![Screenshot](./photos/newplot64.png) |  ![Screenshot](./photos/newplot65.png)  



