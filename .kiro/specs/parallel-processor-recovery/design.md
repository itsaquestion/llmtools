# 设计文档

## 概述

为ParallelLLMProcessor类添加一个`recover_from_database`方法，该方法能够从现有的SQLite数据库文件中恢复未完成的处理任务。该方法将识别数据库中结果为空（NULL）、空字符串或"NA"的记录，重新处理对应的提示词，并更新数据库中的结果。

## 架构

### 方法签名
```python
def recover_from_database(self, db_file_path: str) -> List[str]:
    """
    从指定的数据库文件中恢复未完成的处理任务
    
    Args:
        db_file_path: SQLite数据库文件的路径
        
    Returns:
        List[str]: 按原始顺序排列的完整结果列表
        
    Raises:
        FileNotFoundError: 数据库文件不存在
        ValueError: 数据库格式无效或缺少必要的表
        sqlite3.Error: 数据库操作错误
    """
```

### 处理流程

1. **数据库验证阶段**
   - 检查数据库文件是否存在
   - 验证llm_results表是否存在
   - 检查表结构是否符合预期

2. **数据分析阶段**
   - 读取所有记录并按id排序
   - 识别需要重新处理的记录（result为NULL、空字符串或"NA"）
   - 提取对应的提示词和索引信息

3. **重新处理阶段**
   - 使用当前处理器配置重新处理失败的提示词
   - 显示处理进度
   - 处理过程中的错误处理

4. **数据库更新阶段**
   - 将新的结果更新到数据库对应位置
   - 保持数据完整性
   - 记录更新状态

5. **结果返回阶段**
   - 从数据库读取所有最终结果
   - 按原始顺序返回完整列表

## 组件和接口

### 核心组件

#### 1. DatabaseValidator
负责验证数据库文件和表结构的有效性。

```python
class DatabaseValidator:
    @staticmethod
    def validate_database_file(db_file_path: str) -> None:
        """验证数据库文件是否存在且可访问"""
        
    @staticmethod
    def validate_table_structure(db_file_path: str) -> None:
        """验证llm_results表是否存在且结构正确"""
```

#### 2. RecoveryAnalyzer
分析数据库中的数据，识别需要重新处理的记录。

```python
class RecoveryAnalyzer:
    @staticmethod
    def analyze_database(db_file_path: str) -> Tuple[List[Tuple[int, str]], List[str]]:
        """
        分析数据库，返回需要重新处理的记录和所有现有结果
        
        Returns:
            Tuple[List[Tuple[int, str]], List[str]]: 
            (需要重新处理的(id, prompt)列表, 所有现有结果列表)
        """
        
    @staticmethod
    def is_result_incomplete(result: Any) -> bool:
        """判断结果是否需要重新处理"""
```

#### 3. RecoveryProcessor
执行实际的重新处理逻辑。

```python
class RecoveryProcessor:
    def __init__(self, parallel_processor: 'ParallelLLMProcessor'):
        self.processor = parallel_processor
        
    def process_failed_prompts(self, failed_records: List[Tuple[int, str]]) -> Dict[int, str]:
        """重新处理失败的提示词，返回id到结果的映射"""
```

#### 4. DatabaseUpdater
负责将新结果更新到数据库。

```python
class DatabaseUpdater:
    @staticmethod
    def update_results(db_file_path: str, results_map: Dict[int, str]) -> List[int]:
        """
        更新数据库中的结果
        
        Returns:
            List[int]: 更新失败的记录ID列表
        """
```

### 主要接口

#### recover_from_database方法的详细实现逻辑

```python
def recover_from_database(self, db_file_path: str) -> List[str]:
    # 1. 验证数据库
    DatabaseValidator.validate_database_file(db_file_path)
    DatabaseValidator.validate_table_structure(db_file_path)
    
    # 2. 分析数据库
    failed_records, existing_results = RecoveryAnalyzer.analyze_database(db_file_path)
    
    # 3. 检查是否需要处理
    if not failed_records:
        logger.info("所有结果都已完成，无需恢复处理")
        return existing_results
    
    logger.info(f"发现 {len(failed_records)} 个未完成的记录，开始恢复处理")
    
    # 4. 重新处理失败的提示词
    recovery_processor = RecoveryProcessor(self)
    new_results = recovery_processor.process_failed_prompts(failed_records)
    
    # 5. 更新数据库
    failed_updates = DatabaseUpdater.update_results(db_file_path, new_results)
    if failed_updates:
        logger.warning(f"以下记录更新失败: {failed_updates}")
    
    # 6. 读取并返回最终结果
    _, final_results = RecoveryAnalyzer.analyze_database(db_file_path)
    
    logger.info(f"恢复处理完成，共处理 {len(failed_records)} 个记录")
    return final_results
```

## 数据模型

### 数据库表结构
使用现有的llm_results表结构：

```sql
CREATE TABLE llm_results (
    id INTEGER PRIMARY KEY,           -- 对应prompt列表索引+1
    prompt TEXT NOT NULL,             -- 原始提示词
    result TEXT,                      -- LLM推理结果，初始为NULL
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- 创建时间
);
```

### 内部数据结构

#### FailedRecord
```python
@dataclass
class FailedRecord:
    id: int           # 数据库记录ID
    prompt: str       # 原始提示词
    original_index: int  # 在原始列表中的索引（id-1）
```

#### RecoveryResult
```python
@dataclass
class RecoveryResult:
    total_records: int      # 总记录数
    failed_records: int     # 失败记录数
    recovered_records: int  # 成功恢复的记录数
    failed_updates: List[int]  # 更新失败的记录ID
```

## 错误处理

### 异常类型

1. **FileNotFoundError**: 数据库文件不存在
2. **ValueError**: 数据库格式无效或表结构不正确
3. **sqlite3.Error**: 数据库操作相关错误
4. **RuntimeError**: 处理过程中的运行时错误

### 错误处理策略

1. **数据库验证错误**: 立即抛出异常，提供清晰的错误信息
2. **部分记录处理失败**: 继续处理其他记录，记录失败信息
3. **数据库更新失败**: 记录失败的记录ID，不影响其他更新
4. **并发访问冲突**: 使用重试机制处理数据库锁定

### 日志记录

```python
# 开始恢复
logger.info(f"开始从数据库恢复: {db_file_path}")
logger.info(f"发现 {len(failed_records)} 个未完成记录")

# 处理进度
logger.debug(f"正在处理记录 {record_id}: {prompt[:50]}...")

# 更新结果
logger.debug(f"更新记录 {record_id} 成功")
logger.warning(f"更新记录 {record_id} 失败: {error}")

# 完成恢复
logger.info(f"恢复完成，成功处理 {success_count}/{total_count} 个记录")
```

## 测试策略

### 单元测试

1. **数据库验证测试**
   - 测试不存在的数据库文件
   - 测试无效的数据库格式
   - 测试缺少llm_results表的情况

2. **数据分析测试**
   - 测试识别NULL结果
   - 测试识别空字符串结果
   - 测试识别"NA"结果
   - 测试所有结果都完成的情况

3. **重新处理测试**
   - 测试单个失败记录的处理
   - 测试多个失败记录的批量处理
   - 测试处理过程中的错误处理

4. **数据库更新测试**
   - 测试成功更新记录
   - 测试部分更新失败的情况
   - 测试并发更新的处理

### 集成测试

1. **完整恢复流程测试**
   - 创建包含失败记录的测试数据库
   - 执行恢复操作
   - 验证最终结果的正确性

2. **边界条件测试**
   - 空数据库的处理
   - 所有记录都失败的情况
   - 所有记录都成功的情况

3. **性能测试**
   - 大量失败记录的恢复性能
   - 内存使用情况
   - 数据库操作效率

### 测试数据准备

```python
def create_test_database_with_failures():
    """创建包含失败记录的测试数据库"""
    # 创建测试数据库
    # 插入部分成功和部分失败的记录
    # 返回数据库文件路径和预期结果
```

## 性能考虑

### 内存优化
- 分批读取大型数据库的记录，避免一次性加载所有数据
- 使用生成器处理大量失败记录

### 数据库优化
- 使用事务批量更新结果，减少数据库I/O
- 利用索引优化查询性能
- 使用WAL模式支持并发访问

### 并发处理
- 重用现有的并发处理配置（num_workers等）
- 保持与原始处理器相同的性能特征

## 兼容性

### 向后兼容性
- 不修改现有的数据库表结构
- 不影响现有的处理方法
- 保持现有API的稳定性

### 数据库版本兼容性
- 支持现有版本的数据库格式
- 处理可能的数据类型变化
- 兼容不同SQLite版本的特性差异