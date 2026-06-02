import os
import pickle
from Bio.PDB import PDBParser, NeighborSearch, Selection, DSSP
from Bio.SeqUtils import seq1
import warnings
import numpy as np
import shutil
# from utils_tool import *
warnings.filterwarnings("ignore")

'''
识别抗体表面残基：根据溶剂可及性（使用 DSSP 计算），确定在单独的抗体结构中有哪些氨基酸残基位于蛋白质表面。
提取表面特征：从一个预先存在的、包含抗体所有残基特征的数据文件 (.pkl 文件中的 ab_feature_data) 中，只挑出那些属于表面残基的特征数据。
更新数据集：将这些提取出来的、仅属于表面残基的特征数据，添加（或覆盖）到 .pkl 数据文件中对应的条目里。 得到pecan-paratope-val-all.pkl
脚本中大量关于结合位点识别和邻接关系计算的代码目前处于非活动状态（被注释掉了）
'''
'''
抗体-抗原复合物 (abag_file)
单独的抗体 (ab_file)
单独的抗原 (ag_file)
'''
def calculate_binding_and_surface_sites(abag_file, ab_file, ag_file, ab_feature_data,distance_threshold=4.5, neighbor_distance_threshold=5,rasa_threshold=0.25, dssp_executable='mkdssp'):
    print(f"Processing {abag_file, ab_file, ag_file}")
    parser = PDBParser(QUIET=True)
    abag_structure = parser.get_structure(os.path.basename(abag_file), abag_file)
    ab_structure = parser.get_structure(os.path.basename(ab_file), ab_file)
    ag_structure = parser.get_structure(os.path.basename(ag_file), ag_file)
    abag_model = abag_structure[0]
    ab_model = ab_structure[0]
    ag_model = ag_structure[0]
    #单独的ab和ag
    antibody_chains = list(ab_model.get_chains())
    antigen_chains = list(ag_model.get_chains())
    # complex
    # 假设前两个链是抗体链，剩下的链是抗原链
    chains = list(abag_model.get_chains())
    antibody_chains_complex = []
    antigen_chains_complex = []
    for chain in chains:
        if chain in antibody_chains:
            antibody_chains_complex.append(chain)
        else:
            antigen_chains_complex.append(chain)

    # 获取所有残基
    antibody_residues = [res for chain in antibody_chains for res in chain]
    antigen_residues = [res for chain in antigen_chains for res in chain]
    antibody_residues_complex = [res for chain in antibody_chains_complex for res in chain]
    antigen_residues_complex = [res for chain in antigen_chains_complex for res in chain]
    
    antibody_surface_residues = []
    antigen_surface_residues = []
    ab_feature = []
    ag_feature = []
    # 检查长度是否相等，然后执行下面的代码
    if (len(antibody_residues_complex) == len(antibody_residues)) and (len(antigen_residues_complex) == len(antigen_residues)):
        # 创建 NeighborSearch 对象
        abag_atoms = [atom for atom in abag_model.get_atoms() if not atom.name.startswith('H')]
        abag_ns = NeighborSearch(abag_atoms)
        # ab_atoms = Selection.unfold_entities(ab_model, 'A')
        ab_atoms = [atom for atom in ab_model.get_atoms() if not atom.name.startswith('H')]
        ab_ns = NeighborSearch(ab_atoms)
        # ag_atoms = Selection.unfold_entities(ag_model, 'A')
        ag_atoms = [atom for atom in ag_model.get_atoms() if not atom.name.startswith('H')]
        ag_ns = NeighborSearch(ag_atoms)



        # 初始化标签
        antibody_labels = [0] * len(antibody_residues)
        antigen_labels = [0] * len(antigen_residues)
        antibody_surface_labels = [0] * len(antibody_residues)
        antigen_surface_labels = [0] * len(antigen_residues)


        # 计算 DSSP
        ab_dssp = DSSP(ab_model, ab_file, dssp=dssp_executable)
        ag_dssp = DSSP(ag_model, ag_file, dssp=dssp_executable)

        # 创建残基到索引的映射
        antibody_residue_to_index = {res: i for i, res in enumerate(antibody_residues)}            # 将res作为键，i作为值方便通过res读取索引i
        antigen_residue_to_index = {res: i for i, res in enumerate(antigen_residues)}
        antibody_residue_to_index_complex = {res: i for i, res in enumerate(antibody_residues_complex)}
        antigen_residue_to_index_complex = {res: i for i, res in enumerate(antigen_residues_complex)}
        # res0 = antibody_residues[0]
        # print(  res0 in antibody_residues_complex )
        # print(antibody_residue_to_index[res0])
        # print(antibody_residue_to_index[1])

        # for res in antibody_residues:
        #     print( res in antibody_residue_to_index)

        # 标记表面残基
        for key in ab_dssp.keys():
            chain_id, res_id = key
            res = ab_model[chain_id][res_id]
            i = antibody_residue_to_index[res]
            rasa = ab_dssp[key][3]
            if rasa >= rasa_threshold:
                antibody_surface_labels[i] = 1
        for key in ag_dssp.keys():
            chain_id, res_id = key
            res = ag_model[chain_id][res_id]
            i = antigen_residue_to_index[res]
            rasa = ag_dssp[key][3]
            if rasa >= rasa_threshold:
                antigen_surface_labels[i] = 1
        #加入表面残基的特征
        for i in range(len(antibody_residues)):
            if antibody_surface_labels[i] == 1:
                antibody_surface_residues.append(antibody_residues[i])
                ab_feature.append(ab_feature_data[i])
        # for i in range(len(antigen_residues)):
        #     if antigen_surface_labels[i] == 1:
        #         antigen_surface_residues.append(antigen_residues[i])
        #         ag_feature.append(ag_feature_data[i])
        # antibody_surface_residue_to_index = {res: i for i, res in enumerate(antibody_surface_residues)}
        # antigen_surface_residue_to_index = {res: i for i, res in enumerate(antigen_surface_residues)}

        # antibody_labels_onsurface = [0] * len(antibody_surface_residues)
        # antigen_labels_onsurface = [0] * len(antigen_surface_residues)

        # # 标记邻居节点
        # antibody_adjacency_labels=[]
        # antigen_adjacency_labels=[]
        # for i, antibody_res in enumerate(antibody_surface_residues):
        #     antibody_adjacency_labels_r = [0] * len(antibody_surface_residues)
        #     antibody_adjacency_labels_r[i] = 1
        #     for atom in antibody_res: 
        #         neighbors = ab_ns.search(atom.coord, neighbor_distance_threshold)   #改为邻居节点的阈值
        #         for neighbor in neighbors:
        #             neighbor_residue = neighbor.get_parent()
        #             if neighbor_residue in antibody_surface_residues:
        #                 antibody_neighbor_index = antibody_surface_residue_to_index[neighbor_residue]
        #                 antibody_adjacency_labels_r[antibody_neighbor_index] = 1
        #     antibody_adjacency_labels.append(antibody_adjacency_labels_r)
     

        # for i, antigen_res in enumerate(antigen_surface_residues):
        #     antigen_adjacency_labels_r = [0] * len(antigen_surface_residues)
        #     antigen_adjacency_labels_r[i] = 1
        #     for atom in antigen_res:
        #         neighbors = ag_ns.search(atom.coord, neighbor_distance_threshold)   #改为邻居节点的阈值
        #         for neighbor in neighbors:
        #             neighbor_residue = neighbor.get_parent()
        #             if neighbor_residue in antigen_surface_residues:
        #                 antigen_neighbor_index = antigen_surface_residue_to_index[neighbor_residue]
        #                 # print(antigen_neighbor_index)
        #                 antigen_adjacency_labels_r[antigen_neighbor_index] = 1
        #     antigen_adjacency_labels.append(antigen_adjacency_labels_r)

        # # # 将antibody_adjacency_labels和antigen_adjacency_labels转化为array
        # antibody_adjacency_labels=np.array(antibody_adjacency_labels)
        # antigen_adjacency_labels=np.array(antigen_adjacency_labels)
        # # 标记结合位点
        # for i, antibody_res in enumerate(antibody_residues_complex):
        #     for atom in antibody_res:
        #         neighbors = abag_ns.search(atom.coord, distance_threshold)
        #         for neighbor in neighbors:
        #             neighbor_residue = neighbor.get_parent()
        #             if neighbor_residue in antigen_residue_to_index_complex:
        #                 antibody_labels[i] = 1
        #                 antigen_index = antigen_residue_to_index_complex[neighbor_residue]
        #                 antigen_labels[antigen_index] = 1
        # # 标记表面结合位点
        # for i, antibody_res in enumerate(antibody_residues_complex):
        #     if antibody_labels[i] == 1:     #找到表面的结合位点
        #         antibody_binding_index = antibody_residue_to_index_complex[antibody_res]    #找到结合残基在复合体中的序号=单独ab链序号
        #         antibody_binding_res_in_chain = (list(antibody_residue_to_index.keys()))[antibody_binding_index]             #找到在单独ab链中的结合残基
        #         if antibody_binding_res_in_chain in antibody_surface_residues:
        #             antibody_surface_index = antibody_surface_residue_to_index[antibody_binding_res_in_chain]    #找到在surface_residue中的序号
        #             antibody_labels_onsurface[antibody_surface_index] = 1
        # for i, antigen_res in enumerate(antigen_residues_complex):
        #     if antigen_labels[i] == 1:     #找到表面的结合位点
        #         antigen_binding_index = antigen_residue_to_index_complex[antigen_res]    #找到结合残基在复合体中的序号=单独ag链序号
        #         antigen_binding_res_in_chain = (list(antigen_residue_to_index.keys()))[antigen_binding_index]             #找到在单独ag链中的结合残基
        #         if antigen_binding_res_in_chain in antigen_surface_residues:
        #             antigen_surface_index = antigen_surface_residue_to_index[antigen_binding_res_in_chain]    #找到在surface_residue中的序号
        #             antigen_labels_onsurface[antigen_surface_index] = 1
                    
                    
        # 获取序列
        # antibody_sequence = ''.join(seq1(res.get_resname()) for res in antibody_residues)
        # antigen_sequence = ''.join(seq1(res.get_resname()) for res in antigen_residues)

        # antibody_index=list(antibody_residue_to_index)
        # antigen_index=list(antigen_residue_to_index)
        # antibody_surface_index=list(antibody_surface_residue_to_index)
        # antigen_surface_index=list(antigen_surface_residue_to_index)
        antibody_sequence = ''.join(seq1(res.get_resname()) for res in antibody_residues)
        antigen_sequence = ''.join(seq1(res.get_resname()) for res in antigen_residues)
        return {
            # 'protein_name': os.path.basename(abag_file).replace('.pdb', ''),
            # 'antibody_sequence': antibody_sequence,
            # 'antigen_sequence': antigen_sequence,
            # 'antibody_labels': antibody_labels,
            # 'antigen_labels': antigen_labels,
            # 'antibody_surface_labels': antibody_surface_labels,
            # 'antigen_surface_labels': antigen_surface_labels,
            # 'antibody_labels_onsurface': antibody_labels_onsurface,
            # 'antigen_labels_onsurface': antigen_labels_onsurface,
            # 'antibody_adjacency_labels_onsurface': antibody_adjacency_labels,
            # 'antigen_adjacency_labels_onsurface': antigen_adjacency_labels,
            'surface_ab_feature': np.array(ab_feature),
            # 'surface_ag_feature': np.array(ag_feature)
            # 'antigen_index': antigen_index,
            # 'antibody_index': antibody_index,
            # 'antibody_surface_index': antibody_surface_index,
            # 'antigen_surface_index': antigen_surface_index
        }
    
    # else:
    #     # 如果长度不相等，则输出错误信息或者采取其他处理方式
    #     # print("Error: Antibody or antigen residue lengths do not match between complex and individual chains.")
    #     print(f"wrong files: {abag_file}, {ab_file}, {ag_file}")
    #     error_folder_abag = r'C:\Users\60410\Desktop\新建文件夹 (2)\error_files\wrong_abag'
    #     error_folder_ab = r'C:\Users\60410\Desktop\新建文件夹 (2)\error_files\wrong_ab'
    #     error_folder_ag = r'C:\Users\60410\Desktop\新建文件夹 (2)\error_files\wrong_ag'

    #     os.makedirs(error_folder_abag, exist_ok=True)
    #     os.makedirs(error_folder_ab, exist_ok=True)
    #     os.makedirs(error_folder_ag, exist_ok=True)

    #     error_files = [abag_file, ab_file, ag_file]


    #     shutil.copyfile(error_files[0], os.path.join(error_folder_abag, os.path.basename(error_files[0])))

    #     shutil.copyfile(error_files[1], os.path.join(error_folder_ab, os.path.basename(error_files[1])))

    #     shutil.copyfile(error_files[2], os.path.join(error_folder_ag, os.path.basename(error_files[2])))


    #     print(f"Error files copied to {error_folder_abag}, {error_folder_ab}, {error_folder_ag}")

def process_pdb_files(abag_chain_folder, ab_chain_folder, ag_chain_folder,ori_file):
    abag_chain_files = [os.path.join(abag_chain_folder, f) for f in os.listdir(abag_chain_folder) if f.endswith('.pdb')]
    ab_chain_files = [os.path.join(ab_chain_folder, f) for f in os.listdir(ab_chain_folder) if f.endswith('.pdb')]
    ag_chain_files = [os.path.join(ag_chain_folder, f) for f in os.listdir(ag_chain_folder) if f.endswith('.pdb')]
    #useless = r'C:\Users\60410\Desktop\新建文件夹 (2)\labels-AbAg-pdb-chain-matched-7.16-2.pkl'
    #useless = r'C:\Users\60410\Desktop\make_pecan\pecan.pkl'

    with open(ori_file, 'rb') as f:
        loaded_data = pickle.load(f)
    results = []
    for i,data in enumerate (loaded_data):
    # for i in range(1):  
        sur_pms = calculate_binding_and_surface_sites(abag_chain_files[i], ab_chain_files[i], ag_chain_files[i],data['ab_feature'])
        data['surface_ab_feature'] = sur_pms['surface_ab_feature']
        # data['surface_ag_feature'] = sur_pms['surface_ag_feature']

    
    # 保存到 PKL 文件
    with open(ori_file, 'wb') as f:
        pickle.dump(loaded_data, f)

    print(f"Results saved to {ori_file}")
# for i, data in enumerate(loaded_data):
#     data['ab_feature'] = features[i]

# with open(origin_file, 'wb') as f:
#     pickle.dump(loaded_data, f)
if __name__ == "__main__":
    abag_chain_folder = r'C:\Users\60410\Desktop\make_pecan\pecan-paratope-val'  # 替换为包含 PDB 文件的文件夹路径
    ab_chain_folder=r'C:\Users\60410\Desktop\make_pecan\pecan-paratope-val-ab'
    ag_chain_folder=r'C:\Users\60410\Desktop\make_pecan\pecan-paratope-val-ag'   
    output_file = r'C:\Users\60410\Desktop\make_pecan\pecan-paratope-val-all.pkl'
    # output_file = r'C:\Users\60410\Desktop\新建文件夹 (2)\labels-AbAg-pdb-chain-matched-7.23-onlysurface-with-pmsfeature.pkl'

    process_pdb_files(abag_chain_folder, ab_chain_folder, ag_chain_folder, output_file) 