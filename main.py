import requests
from transformers import AutoTokenizer, AutoModel

# 1. 模型配置
deepseek_api_url = "YOUR_DEEPSEEK_API_URL"  # 替换为DeepSeek API URL
qwen_model_name = "Qwen/Qwen-7B-Chat"  # 或其他Qwen模型名称
local_classifier_model_path = "YOUR_LOCAL_CLASSIFIER_MODEL_PATH"  # 替换为本地总分类模型路径

# 2. 加载模型和tokenizer
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
qwen_model = AutoModel.from_pretrained(qwen_model_name)
# 加载本地总分类模型，您可能需要根据模型类型进行调整
local_classifier_tokenizer = AutoTokenizer.from_pretrained(local_classifier_model_path)
local_classifier_model = AutoModel.from_pretrained(local_classifier_model_path)

# 3. 定义模型调用函数
def call_deepseek_api(prompt):
    """调用DeepSeek API"""
    payload = {"prompt": prompt}
    try:
        response = requests.post(deepseek_api_url, json=payload)
        response.raise_for_status()  # 检查HTTP错误
        return response.json().get("result")
    except requests.exceptions.RequestException as e:
        print(f"DeepSeek API 调用失败: {e}")
        return None

def call_qwen_model(prompt):
    """调用Qwen模型"""
    inputs = qwen_tokenizer(prompt, return_tensors="pt")
    outputs = qwen_model(**inputs)
    # 处理模型输出，提取文本结果
    # 这里需要根据Qwen模型的输出结构进行调整
    return qwen_tokenizer.decode(outputs.last_hidden_state[0], skip_special_tokens=True)

def call_local_classifier_model(prompt):
    """调用本地总分类模型"""
    inputs = local_classifier_tokenizer(prompt, return_tensors="pt")
    outputs = local_classifier_model(**inputs)
    # 处理模型输出，提取分类结果
    # 这里需要根据总分类模型的输出结构进行调整
    return local_classifier_tokenizer.decode(outputs.last_hidden_state[0], skip_special_tokens=True)

# 4. 主函数
def main(user_input):
    """主函数，处理用户输入并调用模型"""
    # 1. 调用总分类模型，确定任务类型
    task_type = call_local_classifier_model(user_input)

    # 2. 根据任务类型，调用相应的模型
    if "nlp" in task_type.lower():  # 假设总分类模型输出包含"nlp"表示自然语言处理任务
        deepseek_result = call_deepseek_api(user_input)
        qwen_result = call_qwen_model(user_input)
        print(f"DeepSeek 结果: {deepseek_result}")
        print(f"Qwen 结果: {qwen_result}")
    elif "image" in task_type.lower():#假设总分类模型输出包含"image"表示图像处理任务
        print("图像处理任务，待实现")
    else:
        print("不支持的任务类型")

# 5. 运行主函数
if __name__ == "__main__":
    user_input = input("请输入您的任务: ")
    main(user_input)
