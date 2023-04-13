pointnet = PointNetSeg()
path='/content/gdrive/MyDrive/level3loss54_62.663.pth'
pointnet.load_state_dict(torch.load(path))#,map_location=torch.device('cpu')))
pointnet.eval()


batch = next(iter(val_loader))
pred = pointnet(batch['image'].transpose(1,2))
pred_np = np.array(torch.argmax(pred[0],1));
acc = (pred_np==np.array(batch['category']))

resulting_acc = np.sum(acc, axis=-1) / 10000
resulting_acc

x,y,z=np.array(batch['image'][1]).T
c = np.array(batch['category'][1]).T
#c = np.array(pred_np[0]).T

fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, 
                                   mode='markers',
                                   marker=dict(
        size=30,
        color=c,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=1.0
    ))])
fig.update_traces(marker=dict(size=2,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
fig.show()