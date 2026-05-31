import pandas as pd

def parse_obo_for_labels(obo_file, label_mapping_file, output_csv):
    """
    从 go-basic.obo 解析指定 GO 标签的父子关系
    """
    # 1. 读取您关心的 112 个 GO 标签（支持 .gz 压缩文件）
    if label_mapping_file.endswith('.gz'):
        label_df = pd.read_csv(label_mapping_file, compression='gzip')
    else:
        label_df = pd.read_csv(label_mapping_file)
    
    go_to_idx = {row['GO']: idx for idx, row in label_df.iterrows()}
    print(f"Loaded {len(go_to_idx)} target GO terms")

    # 2. 解析 obo 文件，提取所有 is_a 关系
    is_a_relations = []
    current_term_id = None
    
    with open(obo_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('[Term]'):
                current_term_id = None
            elif line.startswith('id: GO:'):
                current_term_id = line.split(' ')[1]
            elif line.startswith('is_a: GO:') and current_term_id:
                parent_id = line.split(' ')[1]
                is_a_relations.append((current_term_id, parent_id))
    
    print(f"Found {len(is_a_relations)} total is_a relations in GO")

    # 3. 过滤
    filtered_parents = []
    for child_go, parent_go in is_a_relations:
        if child_go in go_to_idx and parent_go in go_to_idx:
            child_idx = go_to_idx[child_go]
            parent_idx = go_to_idx[parent_go]
            if child_idx != parent_idx:
                filtered_parents.append((child_idx, parent_idx))
    
    filtered_parents = list(set(filtered_parents))
    
    # 4. 保存
    df = pd.DataFrame(filtered_parents, columns=['child_idx', 'parent_idx'])
    df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Saved {len(df)} relationships to {output_csv}")
    print(f"  Children: {df['child_idx'].nunique()}, Parents: {df['parent_idx'].nunique()}")
    
    return df


if __name__ == "__main__":
    OBO_FILE = "/mnt/SGCN/src/go-basic.obo"
    LABEL_FILE = "/mnt/SGCN/dataset/ogbn_proteins/mapping/labelidx2GO.csv.gz"  # 直接读压缩版
    OUTPUT_FILE = "/mnt/SGCN/dataset/ogbn_proteins/mapping/go_parents.csv"
    
    parse_obo_for_labels(OBO_FILE, LABEL_FILE, OUTPUT_FILE)