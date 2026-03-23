import numpy as np
from sklearn.model_selection import train_test_split

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import torch

# 1.1 创建小型知识图谱（简化版三国人物关系）
triples = [
    # 人物关系
    ("刘备", "结义", "关羽"),
    ("刘备", "结义", "张飞"),
    ("关羽", "哥哥", "刘备"),
    ("张飞", "哥哥", "刘备"),

    # 亲属关系
    ("刘备", "妻子", "孙尚香"),
    ("刘备", "儿子", "刘禅"),

    # 敌对关系
    ("刘备", "敌对", "曹操"),
    ("刘备", "盟友", "孙权"),
    ("曹操", "敌对", "孙权"),

    # 地理关系
    ("刘备", "据守", "蜀地"),
    ("曹操", "据守", "魏地"),
    ("孙权", "据守", "吴地"),
]

# 1.2 分割数据集
train_triples, test_triples = train_test_split(triples, test_size=0.2, random_state=42)
print(f"训练集: {len(train_triples)} 个三元组")
print(f"测试集: {len(test_triples)} 个三元组")

# PyKEEN 需要 TriplesFactory 或内置数据集名称，不能直接传 Python 列表
train_tf = TriplesFactory.from_labeled_triples(np.asarray(train_triples, dtype=str))
test_tf = TriplesFactory.from_labeled_triples(
    np.asarray(test_triples, dtype=str),
    entity_to_id=train_tf.entity_to_id,
    relation_to_id=train_tf.relation_to_id,
)

# 2.1 核心训练流程
# 思考：TransE 的核心假设是 h + r ≈ t
# 我们将实体和关系映射到低维向量空间，通过向量运算学习这种平移关系
print("\n=== 开始训练 TransE 模型 ===")
result = pipeline(
    # 输入数据
    training=train_tf,
    testing=test_tf,

    # 模型配置
    model="TransE",
    model_kwargs=dict(
        embedding_dim=50,  # 嵌入维度 - 权衡表达能力和计算复杂度
        scoring_fct_norm=1,  # L1范数 - 控制向量差异的度量方式
    ),

    # 训练配置
    training_kwargs=dict(
        num_epochs=200,  # 训练轮数
        batch_size=32,   # 批量大小
    ),

    # 优化器配置
    optimizer="Adam",
    optimizer_kwargs=dict(lr=0.001),

    random_seed=42,
    device="cuda",  # 可改为 'cuda' 使用GPU加速
)

# 2.2 保存模型
result.save_to_directory("./transe_demo_model")
print("模型训练完成并保存！")

# 3.1 加载训练好的模型
model = result.model
entity_embeddings = model.entity_representations[0]
relation_embeddings = model.relation_representations[0]

# 获取实体和关系的映射
# 直接使用训练时的 TriplesFactory，兼容不同版本 PyKEEN
entity_to_id = train_tf.entity_to_id
relation_to_id = train_tf.relation_to_id
id_to_entity = {v: k for k, v in entity_to_id.items()}
id_to_relation = {v: k for k, v in relation_to_id.items()}


# 3.2 核心推理函数
def predict_relation(head_entity, tail_entity, top_k=3):
    """
    给定头实体和尾实体，预测最可能的关系
    核心思想：寻找最接近 (t - h) 的关系向量
    """
    if head_entity not in entity_to_id:
        print(f"[WARN] 未知头实体: {head_entity}")
        return []
    if tail_entity not in entity_to_id:
        print(f"[WARN] 未知尾实体: {tail_entity}")
        return []

    h_id = entity_to_id[head_entity]
    t_id = entity_to_id[tail_entity]

    # 获取向量表示
    h_vec = entity_embeddings(indices=torch.tensor([h_id]))
    t_vec = entity_embeddings(indices=torch.tensor([t_id]))

    # 计算向量差：t - h
    target_diff = t_vec - h_vec

    # 比较与所有关系向量的距离
    relation_scores = []
    for r_id, r_name in id_to_relation.items():
        r_vec = relation_embeddings(indices=torch.tensor([r_id]))

        # 计算 L1 距离（与训练时一致）
        distance = torch.norm(target_diff - r_vec, p=1, dim=-1)
        relation_scores.append((r_name, distance.item()))

    # 按距离升序排序（距离越小越可能）
    relation_scores.sort(key=lambda x: x[1])
    return relation_scores[:top_k]


# 3.3 链路预测函数
def predict_tail(head_entity, relation, top_k=3):
    """
    给定头实体和关系，预测最可能的尾实体
    核心思想：寻找最接近 (h + r) 的实体向量
    """
    if head_entity not in entity_to_id:
        print(f"[WARN] 未知头实体: {head_entity}")
        return []
    if relation not in relation_to_id:
        print(f"[WARN] 未知关系: {relation}")
        return []

    h_id = entity_to_id[head_entity]
    r_id = relation_to_id[relation]

    h_vec = entity_embeddings(indices=torch.tensor([h_id]))
    r_vec = relation_embeddings(indices=torch.tensor([r_id]))

    # 计算目标位置：h + r
    target_position = h_vec + r_vec

    # 比较与所有实体向量的距离
    entity_scores = []
    for e_id, e_name in id_to_entity.items():
        e_vec = entity_embeddings(indices=torch.tensor([e_id]))
        distance = torch.norm(target_position - e_vec, p=1, dim=-1)
        entity_scores.append((e_name, distance.item()))

    entity_scores.sort(key=lambda x: x[1])
    return entity_scores[:top_k]


# 4.1 示例1：已知实体对，预测可能的关系
print("\n=== 示例1：刘备和刘禅可能是什么关系？ ===")
head, tail = "刘备", "刘禅"
predictions = predict_relation(head, tail, top_k=3)
print(f"实体 {head} 和 {tail} 的可能关系：")
for rel, score in predictions:
    print(f"  - {rel} (距离: {score:.3f})")

# 4.2 示例2：给定实体和关系，预测尾实体
print("\n=== 示例2：谁可能是曹操的儿子？ ===")
head, relation = "曹操", "儿子"
predictions = predict_tail(head, relation, top_k=3)
print(f"{head} 的 {relation} 可能是：")
for entity, score in predictions:
    print(f"  - {entity} (距离: {score:.3f})")

# 4.3 示例3：关系类比推理
print("\n=== 示例3：关系类比（曹操之于刘备，如同？之于孙权）===")
# 思路：计算向量差 (刘备 - 曹操) 然后应用到孙权
cao_vec = entity_embeddings(indices=torch.tensor([entity_to_id["曹操"]]))
liu_vec = entity_embeddings(indices=torch.tensor([entity_to_id["刘备"]]))
sun_vec = entity_embeddings(indices=torch.tensor([entity_to_id["孙权"]]))

# 关系向量：刘备 - 曹操
relation_vector = liu_vec - cao_vec

# 应用到孙权：孙权 + (刘备 - 曹操)
target_position = sun_vec + relation_vector

# 寻找最接近的实体
entity_scores = []
for e_id, e_name in id_to_entity.items():
    e_vec = entity_embeddings(indices=torch.tensor([e_id]))
    distance = torch.norm(target_position - e_vec, p=1, dim=-1)
    entity_scores.append((e_name, distance.item()))

entity_scores.sort(key=lambda x: x[1])
print("\n最可能的类比结果：")
for i, (entity, score) in enumerate(entity_scores[:3], 1):
    print(f"{i}. {entity} (距离: {score:.3f})")

# 5.1 在测试集上评估
evaluation_result = result.metric_results.to_df()
print("\n=== 模型评估结果 ===")
print("关键指标：")


def _read_metric(metric_name):
    """优先从 MetricResults 读取，失败时再尝试 DataFrame 回退。"""
    try:
        value = result.metric_results.get_metric(metric_name)
        if value is not None:
            return float(value)
    except Exception:
        pass

    # 兼容不同版本 pykeen 的 to_df 结构
    try:
        if metric_name in evaluation_result.index:
            row = evaluation_result.loc[metric_name]
            if hasattr(row, "to_dict"):
                row_dict = row.to_dict()
                for candidate_col in ["value", "both.realistic", "arithmetic_mean_rank"]:
                    if candidate_col in row_dict:
                        return float(row_dict[candidate_col])
    except Exception:
        pass
    return None


hits1 = _read_metric("hits_at_1")
hits10 = _read_metric("hits_at_10")
mean_rank = _read_metric("mean_rank")

print(f"1. Hits@1: {hits1:.3f}" if hits1 is not None else "1. Hits@1: N/A")
print(f"2. Hits@10: {hits10:.3f}" if hits10 is not None else "2. Hits@10: N/A")
print(f"3. Mean Rank: {mean_rank:.2f}" if mean_rank is not None else "3. Mean Rank: N/A")

# 5.2 可视化向量空间（2D投影）
try:
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # 获取所有实体向量（若模型在 GPU 上，先转到 CPU 再转 numpy）
    embed_device = model.device
    all_entity_ids = torch.tensor(list(range(len(entity_to_id))), device=embed_device)
    all_entity_vecs = entity_embeddings(all_entity_ids).detach().cpu().numpy()

    # PCA降维到2D
    pca = PCA(n_components=2)
    entity_2d = pca.fit_transform(all_entity_vecs)

    # 绘制
    plt.figure(figsize=(10, 8))
    for i, (entity, _) in enumerate(entity_to_id.items()):
        plt.scatter(entity_2d[i, 0], entity_2d[i, 1], s=100, alpha=0.6)
        plt.text(entity_2d[i, 0] + 0.01, entity_2d[i, 1] + 0.01, entity, fontsize=9)

    plt.title("TransE 实体向量空间（PCA降维）")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.savefig("./entity_vectors.png", dpi=150, bbox_inches="tight")
    print("\n实体向量可视化已保存到 entity_vectors.png")

except ImportError:
    print("\n（如需可视化，请安装 matplotlib 和 scikit-learn）")
