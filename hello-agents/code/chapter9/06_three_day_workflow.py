"""
CodebaseMaintainer 三天工作流演示

完整展示长程智能体在三天内的工作流程:
- 第一天: 探索代码库
- 第二天: 分析代码质量
- 第三天: 规划重构任务
- 一周后: 检查进度
"""

import os
# 配置嵌入模型（三选一）
# 方案一：TF-IDF（最简单，无需额外依赖）
os.environ['EMBED_MODEL_TYPE'] = 'tfidf'
os.environ['EMBED_MODEL_NAME'] = ''  # 重要：必须清空，否则会传递不兼容的参数
# 方案二：本地Transformer（需要: pip install sentence-transformers 和 HF token）
# os.environ['EMBED_MODEL_TYPE'] = 'local'
# os.environ['EMBED_MODEL_NAME'] = 'sentence-transformers/all-MiniLM-L6-v2'
# os.environ['HF_TOKEN'] = 'your_hf_token_here'  # 或使用 huggingface-cli login
# 方案三：通义千问（需要API key）
# os.environ['EMBED_MODEL_TYPE'] = 'dashscope'
# os.environ['EMBED_MODEL_NAME'] = 'text-embedding-v3'
# os.environ['EMBED_API_KEY'] = 'your_api_key_here'

from hello_agents import HelloAgentsLLM
from datetime import datetime
import json
import time

# 导入 CodebaseMaintainer
import sys
sys.path.append('.')
from codebase_maintainer import CodebaseMaintainer


def day_1_exploration(maintainer):
    """第一天: 探索代码库"""
    print("\n" + "=" * 80)
    print("第一天: 探索代码库")
    print("=" * 80 + "\n")

    # 1. 初步探索
    print("### 1. 初步探索项目结构 ###")
    response = maintainer.explore()
    print(f"\n助手总结:\n{response[:500]}...\n")

    # 2. 深入分析某个模块
    print("### 2. 分析数据处理模块 ###")
    response = maintainer.run("请查看 data_processor.py 文件，分析其代码设计")
    print(f"\n助手总结:\n{response[:500]}...\n")

    # 模拟时间流逝
    time.sleep(1)


def day_2_analysis(maintainer):
    """第二天: 分析代码质量"""
    print("\n" + "=" * 80)
    print("第二天: 分析代码质量")
    print("=" * 80 + "\n")

    # 1. 整体质量分析
    print("### 1. 查找所有 TODO 注释 ###")
    response = maintainer.analyze()
    print(f"\n助手总结:\n{response[:500]}...\n")

    # 2. 查看具体问题
    print("### 2. 分析 API 客户端代码 ###")
    response = maintainer.run(
        "请分析 api_client.py 的代码质量，特别是错误处理部分，给出改进建议"
    )
    print(f"\n助手总结:\n{response[:500]}...\n")

    # 模拟时间流逝
    time.sleep(1)


def day_3_planning(maintainer):
    """第三天: 规划重构任务"""
    print("\n" + "=" * 80)
    print("第三天: 规划重构任务")
    print("=" * 80 + "\n")

    # 1. 回顾进度
    print("### 1. 回顾当前进度 ###")
    response = maintainer.plan_next_steps()
    print(f"\n助手总结:\n{response[:500]}...\n")

    # 2. 手动创建详细的重构计划
    print("### 2. 创建详细重构计划 ###")
    maintainer.create_note(
        title="本周重构计划 - Week 1",
        content="""## 目标
完成代码质量改进和 TODO 清理

## 任务清单
- [ ] 实现 data_processor.py 中的数据验证逻辑
- [ ] 添加 api_client.py 的重试和错误处理机制
- [ ] 优化 utils.py 的格式化逻辑
- [ ] 补充 models.py 的用户验证方法

## 时间安排
- 周一: 实现数据验证逻辑
- 周二: 完善错误处理
- 周三-周四: 优化工具函数
- 周五: Code Review 和测试

## 风险
- 新增验证逻辑可能影响现有功能
- 需要充分的单元测试覆盖
""",
        note_type="task_state",
        tags=["refactoring", "week1", "high_priority"]
    )
    print("✅ 已创建详细的重构计划\n")

    # 模拟时间流逝
    time.sleep(1)


def week_later_review(maintainer):
    """一周后: 检查进度"""
    print("\n" + "=" * 80)
    print("一周后: 检查进度")
    print("=" * 80 + "\n")

    # 1. 查看笔记摘要
    print("### 1. 笔记摘要 ###")
    summary = maintainer.note_tool.run({"action": "summary"})
    print("📊 笔记摘要:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()

    # 2. 生成完整报告
    print("### 2. 会话报告 ###")
    report = maintainer.generate_report()
    print("\n📄 会话报告:")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def demonstrate_cross_session_continuity():
    """演示跨会话的连贯性"""
    print("\n" + "=" * 80)
    print("演示跨会话的连贯性")
    print("=" * 80 + "\n")

    # 第一次会话
    print("### 第一次会话 (session_1) ###")
    maintainer_1 = CodebaseMaintainer(
        project_name="demo_codebase",
        codebase_path="./codebase",
        llm=HelloAgentsLLM()
    )

    # 创建一些笔记
    maintainer_1.create_note(
        title="代码质量问题",
        content="发现多处 TODO 注释需要实现，特别是数据验证和错误处理部分",
        note_type="blocker",
        tags=["quality", "urgent"]
    )

    stats_1 = maintainer_1.get_stats()
    print(f"会话1统计: {stats_1['activity']}\n")

    # 模拟会话结束
    time.sleep(1)

    # 第二次会话 (新的会话ID,但笔记被保留)
    print("### 第二次会话 (session_2) ###")
    maintainer_2 = CodebaseMaintainer(
        project_name="demo_codebase",  # 同一个项目
        codebase_path="./codebase",
        llm=HelloAgentsLLM()
    )

    # 检索之前的笔记
    response = maintainer_2.run(
        "我们之前发现了什么代码质量问题？现在应该优先处理哪些？"
    )
    print(f"\n助手回答:\n{response[:300]}...\n")

    stats_2 = maintainer_2.get_stats()
    print(f"会话2统计: {stats_2['activity']}\n")

    # 展示笔记摘要
    summary = maintainer_2.note_tool.run({"action": "summary"})
    print("📊 跨会话笔记摘要:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def demonstrate_tool_synergy():
    """演示三大工具的协同"""
    print("\n" + "=" * 80)
    print("演示三大工具的协同")
    print("=" * 80 + "\n")

    maintainer = CodebaseMaintainer(
        project_name="synergy_demo",
        codebase_path="./codebase",
        llm=HelloAgentsLLM()
    )

    # 1. TerminalTool 发现问题
    print("### 1. TerminalTool 查找 TODO 注释 ###")
    todos = maintainer.execute_command("grep -rn 'TODO' --include='*.py' .")
    print(f"发现的 TODO:\n{todos[:300]}...\n")

    # 2. NoteTool 记录发现
    print("### 2. NoteTool 记录发现 ###")
    maintainer.create_note(
        title="待实现功能清单",
        content=f"通过代码扫描发现以下待实现功能:\n{todos[:500]}",
        note_type="conclusion",
        tags=["todo", "analysis"]
    )
    print("✅ 已记录到笔记\n")

    # 3. MemoryTool 存储关键信息 (通过对话)
    print("### 3. MemoryTool 存储关键信息 ###")
    response = maintainer.run("代码库中有哪些待实现的功能？")
    print(f"助手回答:\n{response[:200]}...\n")

    # 4. ContextBuilder 整合所有信息
    print("### 4. ContextBuilder 整合所有信息 ###")
    response = maintainer.run(
        "基于我们的代码分析，应该优先实现哪些 TODO 功能？"
    )
    print(f"助手回答:\n{response[:300]}...\n")

    # 展示统计信息
    stats = maintainer.get_stats()
    print("📊 工具使用统计:")
    print(f"  - 执行的命令: {stats['activity']['commands_executed']}")
    print(f"  - 创建的笔记: {stats['activity']['notes_created']}")
    print(f"  - 发现的问题: {stats['activity']['issues_found']}")


def main():
    """主函数"""
    print("=" * 80)
    print("CodebaseMaintainer 三天工作流演示")
    print("=" * 80)
    
    print("\n💡 使用我们在 chapter9 创建的示例代码库")
    print("📁 代码库路径: ./codebase")
    print("📦 包含文件: data_processor.py, api_client.py, utils.py, models.py\n")

    # 初始化助手
    maintainer = CodebaseMaintainer(
        project_name="demo_codebase",
        codebase_path="./codebase",
        llm=HelloAgentsLLM()
    )

    # 执行三天工作流
    day_1_exploration(maintainer)
    day_2_analysis(maintainer)
    day_3_planning(maintainer)
    week_later_review(maintainer)

    # 额外演示
    print("\n\n" + "=" * 80)
    print("额外演示")
    print("=" * 80)

    demonstrate_cross_session_continuity()
    demonstrate_tool_synergy()

    print("\n" + "=" * 80)
    print("完整演示结束!")
    print("=" * 80)


if __name__ == "__main__":
    main()
