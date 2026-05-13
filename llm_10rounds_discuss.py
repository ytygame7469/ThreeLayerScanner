"""
10轮LLM讨论 v2：流式API + 精简prompt
使用stream=True边收边显示
"""
import os, json, time, requests, sys

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY = "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"
MODEL = "deepseek-ai/deepseek-v4-pro"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "llm_10rounds")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_PROMPT = "你是A股量化交易系统架构师，精通多因子模型和板块轮动分析。请给出具体可落地的算法方案，包含伪代码。用中文回答，精简。"

ROUNDS = [
    {
        "title": "第1轮：个股强度因子体系优化",
        "prompt": """当前个股强度5因子模型（90天窗口）：价格位置30分、趋势强度25分、动量20分、量价配合15分、涨停检测10分。
请分析：1)还需增加哪些因子（波动率、换手率、连板天数等）？2)权重如何优化？3)如何避免高位钝化（涨5倍还在高位拿满分）？4)大票小票权重是否需要不同？5)给出改进后的因子伪代码。"""
    },
    {
        "title": "第2轮：轮动涨停精确检测算法",
        "prompt": """当前轮动检测只统计板块内不同股票涨停总次数，无时间维度。
真正轮动：同板块不同日期不同股票涨停（非同一只连续涨停），是板块启动前兆。
请设计精确算法：1)时间维度每日统计 2)轮动度=涨停股票更换频率 3)轮动质量评估 4)区分龙头连板vs轮流涨停 5)给出伪代码和评分公式。"""
    },
    {
        "title": "第3轮：强势个股作为板块先行信号",
        "prompt": """用户观点："强势个股有时是板块启动信号"。如地产板块轮流涨停但板块指数未大涨。
若板块出现多只强度>80的个股，但恐贪指数<40，可能是反转信号。
请设计"个股→板块信号传导"框架：1)量化个股领先板块的程度 2)强势股数量阈值 3)区分独立行情vs集体启动 4)历史成功模式特征 5)给出检测伪代码。"""
    },
    {
        "title": "第4轮：90天窗口参数优化",
        "prompt": """当前固定90天窗口。请分析：1)90天vs60/120/250天优劣？2)牛市/熊市/震荡是否需要动态窗口？3)90天max/min是否导致高位股票都接近100分位？4)多窗口加权是否更好(30天0.5+90天0.3+250天0.2)？5)地产轮动场景下窗口选择？6)参数优化方案。"""
    },
    {
        "title": "第5轮：地产板块轮动案例分析框架",
        "prompt": """用户提到地产板块最近总是有轮流涨停的股票。这是底部轮动启动模式：不是龙头连板而是不同股票轮流涨停，资金试探入场。
请设计"底部轮动启动"检测框架：1)区分底部轮动vs高位轮动 2)地产特征：政策驱动、超跌反弹 3)确认条件：涨停数趋势、间隔、量能配合 4)判断轮动是否扩散为板块行情 5)信号等级定义。"""
    },
    {
        "title": "第6轮：假信号过滤与交叉验证",
        "prompt": """已知问题：AI算力上游曾被误判为"恐惧可入场"（平均化陷阱），实际中际旭创涨5倍。
请设计多层过滤：1)成分股位置分布（中位数、分布形态非平均值）2)龙头与跟风乖离率 3)历史准确率回测 4)日线vs周线交叉验证 5)量能验证（放量vs缩量可信度）6)过滤算法伪代码和阈值。"""
    },
    {
        "title": "第7轮：实战交易信号生成",
        "prompt": """将扫描结果转化为实战信号。请设计：1)信号等级S/A/B/卖出定义 2)仓位建议比例 3)动态止损结合恐贪指数 4)信号有效期 5)多信号叠加规则 6)完整信号生成伪代码。S级=低位恐贪+高强势密度+轮动+放量。"""
    },
    {
        "title": "第8轮：多时间框架共振分析",
        "prompt": """设计多周期共振：日线90天(短线)、周线52周(中线)、月线24月(长线)。
共振规则：日强+周涨+月低=最佳买点；日强+周高+月高=追高风险。
请设计：1)各周期具体指标 2)共振强度量化 3)不同模式对应策略 4)完整伪代码。"""
    },
    {
        "title": "第9轮：全市场扫描性能优化",
        "prompt": """当前扫描200只股票，需扩展到全A股5000只。请设计优化方案：1)批量数据获取策略 2)预筛选(成交量前1000) 3)并行计算 4)增量更新(每日只更新新数据) 5)缓存策略 6)优化后架构设计。"""
    },
    {
        "title": "第10轮：最终系统整合方案",
        "prompt": """综合前9轮讨论，给出最终整合方案：1)整体架构图 2)核心算法汇总 3)日报/周报输出格式 4)关键参数配置表 5)每日使用流程 6)与恐贪指数整合方式 7)完整报告示例模板。"""
    },
]


def call_llm_stream(prompt: str, max_tokens: int = 4000) -> str:
    """流式调用LLM"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }

    full_response = []
    print(f"  ⏳ 流式接收中...", flush=True)
    t0 = time.time()

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=600, stream=True)
        if resp.status_code != 200:
            return f"❌ API错误 {resp.status_code}: {resp.text[:300]}"

        chunk_count = 0
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                delta = data["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    full_response.append(content)
                    chunk_count += 1
                    if chunk_count % 20 == 0:
                        print(".", end="", flush=True)
            except json.JSONDecodeError:
                continue

        elapsed = time.time() - t0
        result = "".join(full_response)
        print(f"\n  ✅ 完成 ({elapsed:.0f}秒, {len(result)}字符)", flush=True)
        return result

    except Exception as e:
        print(f"\n  ⚠️ 流式异常: {e}", flush=True)
        if full_response:
            return "".join(full_response)
        return f"❌ 调用失败: {e}"


def main():
    print("=" * 80, flush=True)
    print("  10轮LLM讨论 v2：完善三层扫描系统", flush=True)
    print("=" * 80, flush=True)
    print(f"  输出: {OUTPUT_DIR}", flush=True)
    print(flush=True)

    all_results = {}

    for i, round_info in enumerate(ROUNDS):
        print(f"\n{'='*60}", flush=True)
        print(f"  {round_info['title']}", flush=True)
        print(f"{'='*60}", flush=True)

        round_file = os.path.join(OUTPUT_DIR, f"round_{i+1:02d}_{round_info['title'].replace('：','_').replace(' ','_')}.txt")

        if os.path.exists(round_file):
            with open(round_file, "r", encoding="utf-8") as f:
                existing = f.read()
            if len(existing) > 200 and "调用失败" not in existing and "API错误" not in existing:
                print(f"  ⏭️  跳过（已完成）", flush=True)
                all_results[f"round_{i+1}"] = {"title": round_info["title"], "file": round_file}
                continue

        # 调用LLM（最多重试2次）
        response = None
        for attempt in range(2):
            response = call_llm_stream(round_info["prompt"])
            if response and not response.startswith("❌") and not response.startswith("⚠️"):
                break
            if attempt < 1:
                print(f"  🔄 重试...", flush=True)
                time.sleep(5)

        if not response:
            response = "❌ 所有重试均失败"

        # 保存
        with open(round_file, "w", encoding="utf-8") as f:
            f.write(f"{'='*60}\n{round_info['title']}\n{'='*60}\n\n{response}")

        all_results[f"round_{i+1}"] = {"title": round_info["title"], "file": round_file}
        print(f"  📄 已保存: {os.path.basename(round_file)}", flush=True)

        if i < len(ROUNDS) - 1:
            print(f"  ⏸️  等待3秒...", flush=True)
            time.sleep(3)

    # 汇总
    summary_file = os.path.join(OUTPUT_DIR, "00_讨论汇总.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("10轮LLM讨论汇总\n" + "=" * 60 + "\n\n")
        for key, info in all_results.items():
            f.write(f"## {info['title']}\n文件: {os.path.basename(info['file'])}\n\n")

    print(f"\n{'='*60}", flush=True)
    print(f"  ✅ 10轮讨论完成！结果: {OUTPUT_DIR}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()