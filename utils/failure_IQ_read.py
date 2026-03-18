"""
    读取failure IQ数据集,并返回pd
"""

import pandas as pd
import re

class failure_IQ_read:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def _extract_all_questions_data(self):
        df = pd.read_csv(self.csv_path)
        results = []
        
        for _, row in df.iterrows():
            # 处理正确答案
            correct_str = row['correct'].strip("[]")
            correct_values = [val.strip() == 'True' for val in correct_str.split()]

            # 更精确地处理选项（处理带空格的选项名称）
            options_str = row['options'].strip("[]").replace("'", "")
            
            # 使用正则表达式正确分割选项
            # 匹配被单引号包围的字符串
            pattern = r"'([^']+)'"
            options = re.findall(pattern, row['options'])
            
            # 如果正则匹配失败，回退到原来的分割方法
            if not options:
                # 将连续的空格替换为单个空格，然后按空格分割
                options = [opt for opt in options_str.split() if opt]
            # 构建选项-答案映射字典
            option_answer_dict = dict(zip(options, correct_values))
            
            results.append({
                'id': row['id'],
                'question': row['question'],
                'options': options,
                'answer_mapping': option_answer_dict
            })
        
        return results
    
    def get_data(self):
        """
        读取failure IQ数据集
        
        Returns:
            DataFrame: 包含问题、选项和答案映射的DataFrame
                其结构如下：
                | id | question | options | answer_mapping |
                |----|----------|---------|----------------|
                | 1  | 问题文本   | [选项1,选项2,...] | {选项1: True/False, 选项2: True/False, ...} |
        """
        return self._extract_all_questions_data()