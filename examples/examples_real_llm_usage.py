import time

def mock_chat_fn(prompt: str) -> str:
    """Mock LLM function for testing"""
    time.sleep(0.5)  # Simulate API delay
    return f"Response to: {prompt}"

# Test prompts
test_prompts = [
    "1. Tell me a little story about AI.", # 一个时间较长的输出（最后完结），以检验数据库记录的顺序是否正确
    "2. What color is a ripe banana?",
    "3. What is the chemical symbol for gold?",
    "4. How many legs does a spider have?",
    "5. What is the capital city of Japan?"
]

from llmtools import chat, ParallelLLMProcessor
from functools import partial

chat_gpt = partial(chat,model = 'deepseek/deepseek-chat-v3-0324')

# Initialize and run processor
processor = ParallelLLMProcessor(chat_fn=chat_gpt, num_workers=3, 
                                 save_to_db=True,db_filename='test_db.db')
results = processor.process_prompts(test_prompts)
# print(results)

# TODO: 读取并打印数据库test_db.db的内容
print("\n" + "="*60)
print("数据库内容 (处理完成后):")
print("="*60)

import sqlite3
with sqlite3.connect('test_db.db') as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT id, prompt, result, created_at FROM llm_results ORDER BY id")
    records = cursor.fetchall()
    
    for record_id, prompt, result, created_at in records:
        print(f"ID: {record_id}")
        print(f"提示: {prompt}")
        print(f"结果: {result}")
        print(f"创建时间: {created_at}")
        print("-" * 40)

# TODO: 删除随机2个result，使用recover恢复，然后检查test_db.db中数据的完整性
print("\n" + "="*60)
print("模拟故障恢复测试:")
print("="*60)

# 随机删除2个结果 (设置为NULL来模拟失败)
import random
with sqlite3.connect('test_db.db') as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM llm_results")
    all_ids = [row[0] for row in cursor.fetchall()]
    
    # 随机选择2个ID进行"故障"模拟
    failed_ids = random.sample(all_ids, 2)
    print(f"模拟记录 {failed_ids} 处理失败...")
    
    # 将这些记录的结果设置为NULL
    for record_id in failed_ids:
        cursor.execute("UPDATE llm_results SET result = NULL WHERE id = ?", (record_id,))
    conn.commit()

print("\n故障后的数据库状态:")
with sqlite3.connect('test_db.db') as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT id, result FROM llm_results ORDER BY id")
    records = cursor.fetchall()
    
    for record_id, result in records:
        status = "✓ 完整" if result is not None else "✗ 缺失"
        print(f"记录 {record_id}: {status}")

# 使用recover功能恢复失败的记录
print(f"\n开始恢复失败的记录...")
recovered_results = processor.recover_from_database('test_db.db')

print(f"\n恢复完成! 共处理 {len(recovered_results)} 条记录")

# 检查恢复后的数据完整性
print("\n" + "="*60)
print("恢复后的数据完整性检查:")
print("="*60)

with sqlite3.connect('test_db.db') as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT id, prompt, result FROM llm_results ORDER BY id")
    records = cursor.fetchall()
    
    complete_count = 0
    incomplete_count = 0
    
    for record_id, prompt, result in records:
        if result is not None and result != "" and result != "NA":
            complete_count += 1
            status = "✓ 完整"
        else:
            incomplete_count += 1
            status = "✗ 仍然缺失"
        
        print(f"记录 {record_id}: {status}")
        if record_id in failed_ids:
            print(f"  -> 这是之前失败的记录，现在状态: {status}")

print(f"\n最终统计:")
print(f"完整记录: {complete_count}")
print(f"不完整记录: {incomplete_count}")
print(f"数据完整性: {(complete_count / len(records) * 100):.1f}%")

if incomplete_count == 0:
    print("🎉 所有记录都已成功恢复!")
else:
    print(f"⚠️  仍有 {incomplete_count} 条记录未能恢复")

# 关闭处理器
processor.close()
