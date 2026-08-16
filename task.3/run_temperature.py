# -*- coding: utf-8 -*-
"""
Temperature 敏感性实验（参数稳健性）。

设计：固定 Prompt v3.0、固定模型、固定样本，仅改变 temperature ∈ {0, 0.3, 0.7, 1.0}，
      每档重复 K 次，比较 准确率 / 幻觉率 / 输出一致率 / JSON有效率。

注意：temperature 是 API 采样层参数，本次课程实验中大模型角色由对话内模型扮演，
      无法设定该参数，因此本维度未取得实测数据。接入真实 API 后直接运行本脚本即可补齐。

用法：
    export LLM_API_KEY=sk-xxx
    python3 experiments/run_temperature.py --repeats 5
"""
import argparse, json, os, subprocess, sys, itertools

TEMPS = [0.0, 0.3, 0.7, 1.0]
SUBSET = "J01,J03,J04,J06,J09,J13,J17,J19"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--model", default="gpt-4o-mini")
    a = ap.parse_args()

    for t in TEMPS:
        for k in range(1, a.repeats + 1):
            run = "T%s_r%d" % (str(t).replace(".", ""), k)
            cmd = [sys.executable, "experiments/run_llm.py",
                   "--version", "v3", "--run", run,
                   "--temperature", str(t), "--model", a.model,
                   "--subset", SUBSET]
            print(">>", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=False)

    # 汇总：每个 temperature 档位下，把 K 次重复喂给 consistency 逻辑
    print("\n完成采集。请依次执行：")
    for t in TEMPS:
        tag = str(t).replace(".", "")
        print("  python3 eval/evaluate.py T%s_r1        # 该档位准确率/幻觉率" % tag)
    print("  python3 eval/consistency.py --runs T00_r1,T00_r2,...  # 该档位输出一致率")
    print("\n分析表模板：")
    print("| temperature | 字段准确率 | 幻觉率 | 输出一致率 | JSON有效率 |")
    for t in TEMPS:
        print("| %.1f | | | | |" % t)


if __name__ == "__main__":
    main()
