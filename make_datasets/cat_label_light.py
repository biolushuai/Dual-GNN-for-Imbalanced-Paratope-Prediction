import os
import pickle
# from adjacency_to_edge_index  import adjacency_to_edge_index

output_file =  r'C:\Users\60410\Desktop\makedataset\val_data-7.23-onlysurface-with-pmsfeature.pkl'
output_file2 =  r'C:\Users\60410\Desktop\新建文件夹 (2)\labels-AbAg-pdb-chain-matched-7.16-2.pkl'
output_file3 =  r'C:\Users\60410\Desktop\新建文件夹 (2)\labels-AbAg-pdb-chain-matched-8.24-onlysurface-with-pmsfeature-noH.pkl'
# output_file4 = r'C:\Users\60410\Desktop\make_pecan\pecan-epitope-val.pkl'
# output_file4 = r'C:\Users\60410\Desktop\make_pecan\pecan-epitope-val-finetuned_on_pecantrain.pkl'
output_file4 = r'data\pecan-paratope-train-all.pkl'

'''
当前使用的是 r'data\pecan-paratope-train-all.pkl'。
这段代码的主要目的是加载一个预先处理好的数据集（存储在 .pkl 文件中），并对该数据集进行两项主要操作：
数据结构检查/探索： 查看并打印数据集中第一个样本（通常是列表中的第一个元素）包含哪些键（key），以及每个键对应的值（value）是什么类型、什么形状（如果是数组或张量）。
数据统计计算： 遍历整个数据集，统计其中与抗体结合位点相关的标签（存储在 'antibody_labels' 键下）中，标签为 0 和标签为 1 的总数。
'''

# output_file4 = r'D:\ThesisCode\Datasets\paratope1024\pecan-paratope-train.pkl'

with open(output_file4, 'rb') as f:
    loaded_results = pickle.load(f)
# print(len(loaded_results))
# print(len(loaded_results['sage_x']))
# Print the loaded results
    # for k, v in train_data[0].items():                                # k,v: key, value
    #     if k != 'PDBID':
    #         print(k, v.shape)
for k,v in loaded_results[0].items():
    print(k)
    print(type(v))
    if hasattr(v, 'shape'):
        if v.shape == (1,):
            pass
        print(v.shape)
# print(loaded_results[0]['antibody_adjacency_labels'])
# print(loaded_results[1]['antibody_adjacency_labels_onsurface'].shape)
# print(type(loaded_results[1]['antibody_adjacency_labels_onsurface']))
# a=adjacency_to_edge_index(loaded_results[1]['antibody_adjacency_labels_onsurface'])
# print(a)
# print(a.shape)
print((loaded_results[0]['antibody_surface_index']))
print(len(loaded_results[0]['antibody_surface_index']))

# print(type(loaded_results[1]['antibody_adjacency_labels'][0]))
# print(loaded_results[1]['antibody_adjacency_labels'])
# print(len(loaded_results[1]['antibody_surface_labels']))
# print(loaded_results[7]['ab_feature'].shape)
# print(len(loaded_results[7]['antibody_surface_labels']))
# print(loaded_results[7]['antibody_labels_onsurface'])

# # Iterate over the loaded_results

# #计算表面残基比例
# count_surface_labels_0=0
# count_surface_labels_1=0
# for result in loaded_results:
#     antigen_labels = result['antigen_labels']
#     surface_labels = result['antigen_surface_labels']
#     # print(len(antibody_labels))
#     for i in range (len(surface_labels)):
#         if antigen_labels[i]==1:
#             if surface_labels[i]==0:
#                 count_surface_labels_0+=1
#             else:
#                 count_surface_labels_1+=1
# print(count_surface_labels_0)
# print(count_surface_labels_1)

# 计算结合位点比例
count_binding_labels_0=0
count_binding_labels_1=0

for result in loaded_results:
    binding_labels = result['antibody_labels']
    # binding_labels = result['antibody_labels_onsurface']
    # print(len(antibody_labels))
    for i in range (len(binding_labels)):
            if binding_labels[i]==0:
                count_binding_labels_0+=1
            else:
                count_binding_labels_1+=1
print(count_binding_labels_0)
print(count_binding_labels_1)


# count_surface_labels_0=0
# count_surface_labels_1=0

# surface_labels = loaded_results[0]['antibody_labels_onsurface']
# # print(len(antibody_labels))
# for i in range (len(surface_labels)):
#     # if antigen_labels[i]==0:
#         if surface_labels[i]==0:
#             count_surface_labels_0+=1
#         else:
#             count_surface_labels_1+=1
# print(count_surface_labels_0)
# print(count_surface_labels_1)