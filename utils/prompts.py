"""
    把一些调试语句放到一起, 方便调试
"""

import os
import re
import json

PROMPT_TEMPLATES = dict()

PROMPT_TEMPLATES["question_with_context"] = """
你是一个专业的问答系统，以下是一个问题和选项，请你分析并给出正确答案：

问题: {question}

选项: {options}

上下文:{context}

请你根据问题,选项以及上下文，分析并给出正确答案，若是上下文无法提供足够判断的依据，
那么你可以结合自身的知识做出回答。

答案应该是选项中的一个。

请按如下格式返回JSON结果:
{{
    "answer": "选项文本",
    "questionBackground": "对问题背景的理解与分析",
    "entityInfo": {{
        "entity1": "问题与选项中关键实体1的相关核心信息",
        "entity2": "问题与选项中关键实体2的相关核心信息",
        "...": "..."
    }},
    "reasoning": "你的简要推理过程和分析，说明你是如何得出答案的"
}}
"""

PROMPT_TEMPLATES["knowledge_graph_extraction"] = """
你是专家信息提取器和结构化数据组织者。
你的任务是分析提供的文本，尽可能多地提取有价值的实体、属性和关系。

指南：
1. 优先使用以下预定义模式：{schema}
2. 灵活性：如果上下文不符合预定义模式，则根据需要提取有价值的知识
3. 简洁性：属性和三元组应互补，无语义冗余
4. 不要错过任何有用信息
5. 模式演化:如果发现重要的新类型,添加到new_schema_types
6. 时间戳:为每个实体、属性和关系分配一个时间戳,表示和他们有关的时间,格式为"YYYY.MM.DD",
    如果没有明确的时间,可以根据上下文推断一个合理的时间,或者是直接置为"unknown"
7. 提取时不用翻译,保持原文,但在返回的JSON中,属性和关系的命名要尽量规范化,去掉冗余词汇,保留核心语义
8. 只返回合法JSON,不要附加解释、注释、Markdown代码块或其他额外文本

文本：{chunk}

有如下样例:
{{
    "attributes": {{
        "电压传感器": ["生产于2005年", "型号为X100"]
    }},
    "triples": [
        ["电压传感器", "安装在", "变电站A"]
    ],
    "entity_types": {{
        "电压传感器": "device",
        "变电站A": "location"
    }},
    "entity_timeStamps": {{
        "电压传感器": "2005.01.01",
        "生产于2005年": "2005.01.01",
        "安装在": "2005.01.01"
    }},
    "new_schema_types": {{
        "nodes": ["device"],
        "relations": ["安装在"],
        "attributes": ["生产于", "型号为"]
    }}
}}

按照样例返回JSON格式:
{{
    "attributes": {{
        "实体": ["属性"]
    }},
    "triples": [
        ["实体1", "关系", "实体2"]
    ],
    "entity_types": {{
        "实体": "类型"
    }},
  "entity_timeStamps": {{
        "实体": "YYYY.MM.DD 或 unknown",
        "属性": "YYYY.MM.DD 或 unknown",
        "关系": "YYYY.MM.DD 或 unknown"
    }},
    "new_schema_types": {{
        "nodes": [],
        "relations": [],
        "attributes": []
    }}
}}
"""

PROMPT_TEMPLATES["question_decomposition_by_schema"] = """
你是一个专业的问题分解大师，能够将复杂的问题分解成更小、更易处理的子问题。
你还可以根据预定义的模式，识别问题中的实体、关系和属性，并将它们组织成一个清晰的结构。

指南：
1. 优先使用以下预定义模式：{schema}
2. 灵活性：如果问题不完全符合预定义模式，则根据需要识别有价值的实体、关系和属性
3. 简洁性：子问题应互补，无语义冗余
4. 不要错过任何有用信息
5. 模式演化:如果发现重要的新类型,添加到new_schema_types
7. 分解时不用翻译,保持原文,但在返回的JSON中,属性和关系的命名要尽量规范化,去掉冗余词汇,保留核心语义
8. 只返回合法JSON,不要附加解释、注释、Markdown代码块或其他额外文本
9. 对于每个子问题，给出其对应的时间戳，表示和问题相关的时间，格式为"YYYY.MM.DD"，如果没有明确的时间，
可以根据上下文推断一个合理的时间，或者是直接置为"unknown"
10. 每个问题可以被分解为1-5个子问题，过多或过少都可能不是最优的分解
11. 可以尝试将多个实体、关系或属性组合成一个子问题，以减少子问题的数量，但要确保每个子问题都足够清晰和具体

问题：{question}

例如对于问题:
    "In electric motor, when rotor windings fault occurs, 
    which sensor from the choices is most critical in detecting the occurrence of the failure event?"

可以分解成如下子问题:
{{
    "sub_questions": {{
        "What is a rotor windings fault in an electric motor?" : "unknown",
        "What sensors are typically used for motor fault detection?" : "unknown",
        "How do different sensors respond to a rotor windings fault?" : "unknown",
        "Which sensor is most effective for detecting the specific symptoms of a rotor windings fault?" : "unknown"
    }}
}}

需要返回的JSON格式如下:
{{
    "sub_questions": {{
        "子问题1": "YYYY.MM.DD 或 unknown",
        "子问题2": "YYYY.MM.DD 或 unknown",
        ...
    }}
}}
"""

QUERY_TEMPLATES = dict()

QUERY_TEMPLATES["batch_insert_nodes"] = """
// 批量插入节点（去重）
CALL apoc.periodic.iterate(
  "UNWIND $payload.nodes AS n RETURN n",
  "
  WITH n,
       CASE 
         WHEN n.type IS NULL THEN ['UnknownLabel'] 
         ELSE [n.type] 
       END AS labels
  CALL apoc.merge.node(
    labels,
    {name: n.name},
    {timeStamp: coalesce(n.timeStamp, 'unknown'), level: coalesce(n.level, 2)},
    {}
  ) YIELD node
  RETURN count(node)
  ",
  {batchSize: 1000, parallel: true, params: {payload: $payload}}
);
"""

QUERY_TEMPLATES["batch_insert_relationships"] = """
// 批量插入关系（同时兜底创建缺失节点）
CALL apoc.periodic.iterate(
  "UNWIND $payload.rels AS r RETURN r",
  "
  MERGE (s:Entity {name: r.source})
  ON CREATE SET s.type = 'unknown', s.timeStamp = 'unknown', s.level = 2

  MERGE (t:Entity {name: r.target})
  ON CREATE SET
    t.type = CASE WHEN r.relation = 'HAS_ATTRIBUTE' THEN 'attribute' ELSE 'unknown' END,
    t.timeStamp = coalesce(r.timeStamp, 'unknown'),
    t.level = CASE WHEN r.relation = 'HAS_ATTRIBUTE' THEN 1 ELSE 2 END

  CALL apoc.merge.relationship(
    s,
    CASE
      WHEN r.relation IS NULL OR trim(r.relation) = '' THEN 'RELATED'
      ELSE trim(r.relation)
    END,
    {},  // identProps
    {timeStamp: coalesce(r.timeStamp, 'unknown')}, // onCreate
    t,
    {timeStamp: coalesce(r.timeStamp, 'unknown')}  // onMatch
  ) YIELD rel
  RETURN count(rel)
  ",
  {batchSize: 1000, parallel: false, params: {payload: $payload}}
);
"""

QUERY_TEMPLATES["query_by_id"] = """
MATCH (n) WHERE id(n) = $id RETURN n
"""

QUERY_TEMPLATES["get_all_nodes"] = """
MATCH (n) RETURN n
"""