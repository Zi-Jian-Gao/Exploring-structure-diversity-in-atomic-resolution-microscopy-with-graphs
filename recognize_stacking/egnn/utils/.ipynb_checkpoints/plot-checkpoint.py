import dgl
import matplot.pyplot as plt
import networkx as nx

def plot_graph(graph):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    
    edges_data = data.edge_index
    y = data.y
    src = edges_data[0].numpy()
    dst = edges_data[1].numpy()
    g = dgl.graph((src, dst))

    nx_g = g.to_networkx().to_undirected()
    pos = nx.kamada_kawai_layout(nx_g)
    nx.draw(nx_g, pos, with_labels=True, node_color=y)


# def plot_conn_with_img(json_path, dis_thres=35, max_ner=3, light_thres=120):
#     colors = ['green', 'red', 'blue']
    
#     x, y, edge_index, edge_attr, points, img, labels = get_con(json_path, dis_thres, max_ner, light_thres)
#     lbl_sets = list(set(labels.flatten()))
    
#     sp = points[edge_index[0].numpy().astype(np.int32)]
#     tp = points[edge_index[1].numpy().astype(np.int32)]
    
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(img, cmap='gray')
#     for i in range(len(sp)):
#         plt.plot([sp[i][1], tp[i][1]], [sp[i][0], tp[i][0]], linewidth=1, c='y')
        
#     for i in lbl_sets:
#         plt.scatter(points[labels == i][:, 1], points[labels == i][:, 0], c=colors[i-1], s=4)
#         plt.axis('off')
#     plt.subplot(1, 2, 2)
#     plt.imshow(img, cmap='gray')
#     plt.axis('off')

# json_path = '../data/final/slide/img/0.json'
# plot_conn_with_img(json_path)


# json_path = '../data/final/slide/img/0.json'
# x, y, edge_index, edge_attr, _, _, _ = get_con(json_path)
# data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
# plot_graph(data)