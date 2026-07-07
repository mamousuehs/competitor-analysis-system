# main.py
import os
from dotenv import load_dotenv

# 加载根目录 .env 环境变量（密钥全程隐式读取，代码不出现明文）
load_dotenv()

# 导入模块内的功能函数
from modules.rag_chain import build_rag_chain
from modules.data_loader import init_and_fill_vector_db


def main():
    # 向量库统一存放在项目根目录
    vector_db_path = "./chroma_db"

    # 自动检测：没有向量库就先执行灌库，不用单独跑 data_loader
    if not os.path.exists(vector_db_path):
        print("⚠️  未检测到向量库，开始执行灌库...")
        init_and_fill_vector_db(persist_dir=vector_db_path)

    # 构建完整 RAG 查询链路
    print("\n🚀 正在构建RAG查询链路...")
    rag_chain = build_rag_chain(persist_dir=vector_db_path)

    # 执行测试查询
    query = "竞品A最近有什么价格变动？"
    print(f"\n🔍 查询问题：{query}")
    print("⏳ 正在生成结果...")

    result = rag_chain.invoke(query)

    print("\n✅ 查询完成，结构化结果如下：")
    print(result)


if __name__ == "__main__":
    main()