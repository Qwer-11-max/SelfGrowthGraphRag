import ollama

# 基本聊天
stream = ollama.chat(
    model='qwen3.5:9b',
    messages=[
        {
            'role': 'user',
            'content': '你好，能介绍一下自己吗？',
        },
    ],
    stream=True,  # 开启流式输出
)


# 迭代输出流式结果
for response in stream:
    # Ollama 流式响应块中，增量文本位于 message.content
    try:
        content = response['message']['content']
    except KeyError:
        content = ''
    if content:
        print(content, end='')

print()