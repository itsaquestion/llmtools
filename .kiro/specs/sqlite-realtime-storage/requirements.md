# Requirements Document

## Introduction

为ParallelLLMProcessor类添加实时SQLite数据库存储功能，允许用户选择是否将推理过程和结果保存到数据库中。该功能将在每次获得LLM推理结果时实时保存到数据库，确保数据的持久化和可追溯性。

## Requirements

### Requirement 1

**User Story:** 作为开发者，我希望能够选择是否将ParallelLLMProcessor的推理结果保存到SQLite数据库中，以便后续分析和追溯。

#### Acceptance Criteria

1. WHEN 初始化ParallelLLMProcessor时 THEN 系统 SHALL 接受一个布尔参数save_to_db来控制是否保存到数据库
2. WHEN save_to_db为False时 THEN 系统 SHALL 不执行任何数据库操作，保持原有功能不变
3. WHEN save_to_db为True时 THEN 系统 SHALL 启用数据库保存功能

### Requirement 2

**User Story:** 作为开发者，我希望能够指定SQLite数据库文件名，以便将数据保存到指定位置。

#### Acceptance Criteria

1. WHEN 初始化ParallelLLMProcessor时 THEN 系统 SHALL 接受一个字符串参数db_filename来指定数据库文件名
2. IF db_filename未提供 THEN 系统 SHALL 使用默认文件名格式"llm_results_{timestamp}.db"，其中timestamp为当前时间戳
3. WHEN 指定数据库文件名时 THEN 系统 SHALL 在指定路径创建或连接到该数据库文件

### Requirement 3

**User Story:** 作为开发者，我希望数据库有标准化的表结构，以便存储和查询推理数据。

#### Acceptance Criteria

1. WHEN 数据库被创建时 THEN 系统 SHALL 创建一个名为"llm_results"的表
2. WHEN 创建表时 THEN 系统 SHALL 包含以下列：
   - id: INTEGER PRIMARY KEY (序号，对应prompt列表中的索引+1)
   - prompt: TEXT NOT NULL (输入的提示词)
   - result: TEXT (LLM的推理结果，初始为NULL)
   - created_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP (创建时间)
3. IF 表已存在 THEN 系统 SHALL 复用现有表结构

### Requirement 4

**User Story:** 作为开发者，我希望在初始化数据库时就预先填充所有prompt记录，确保数据库记录顺序与prompt列表顺序完全一致。

#### Acceptance Criteria

1. WHEN 开始处理prompt列表时 THEN 系统 SHALL 在数据库初始化阶段一次性插入所有prompt记录
2. WHEN 插入初始记录时 THEN 系统 SHALL 设置id为prompt在列表中的索引+1，prompt字段为对应的提示词，result字段为NULL
3. WHEN 创建记录时 THEN 系统 SHALL 确保数据库记录的顺序与输入prompt列表的顺序完全一致

### Requirement 5

**User Story:** 作为开发者，我希望每次获得LLM推理结果时能够实时更新到数据库中，确保数据的及时性。

#### Acceptance Criteria

1. WHEN 单个prompt处理完成时 THEN 系统 SHALL 根据prompt在原始列表中的索引位置更新对应数据库记录的result字段
2. WHEN 处理失败时 THEN 系统 SHALL 将错误信息保存到对应索引位置记录的result字段中
3. WHEN 更新数据库时 THEN 系统 SHALL 使用prompt的原始索引作为唯一标识符来定位正确的数据库记录
4. WHEN 更新数据库时 THEN 系统 SHALL 确保数据库操作不影响原有的并发处理性能

### Requirement 6

**User Story:** 作为开发者，我希望数据库操作具有良好的错误处理机制，确保数据库问题不会影响主要功能。

#### Acceptance Criteria

1. WHEN 数据库操作失败时 THEN 系统 SHALL 记录错误日志但不中断主要处理流程
2. WHEN 数据库连接失败时 THEN 系统 SHALL 继续执行推理处理，仅跳过数据库保存
3. IF 数据库操作出现异常 THEN 系统 SHALL 提供有意义的错误信息用于调试