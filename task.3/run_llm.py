# -*- coding: utf-8 -*-
"""
真实 API 批量调用驱动。本次实验中大模型角色由对话内模型扮演，
本脚本用于在接入真实 API 后完整复现全部指标（含无法在本地测得的运行时间与 temperature 实验）。

用法：
    export LLM_API_KEY=sk-xxx
    python3 experiments/run_llm.py --version v3 --run run1 --temperature 0.0
    python3 eval/evaluate.py run1

输出：outputs/{run}_{version}_raw.json（键为 sample_id，值为模型原始文本）
      outputs/{run}_{version}_timing.json（逐条真实运行时间与 token 用量）
"""
import argparse, json, os, time, sys

PROMPT_FILE = {"v1": "prompts/v1_zeroshot.txt",
               "v2": "prompts/v2_role_schema.txt",
               "v3": "prompts/v3_fewshot_selfcheck.txt"}


def call_llm(prompt, model, temperature):
    """按所用厂商 SDK 替换此函数即可。以下为 OpenAI 兼容接口示例。"""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["LLM_API_KEY"],
                    base_url=os.environ.get("LLM_BASE_URL"))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=1.0,
        seed=42,                 # 支持 seed 的服务端可进一步降低随机性
        max_tokens=1500,
    )
    usage = resp.usage
    return resp.choices[0].message.content, usage.prompt_tokens, usage.completion_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["v1", "v2", "v3"])
    ap.add_argument("--run", default="run1")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--sample-file", default="data/sample20.json")
    ap.add_argument("--subset", default="")        # 逗号分隔的 sample_id，用于稳定性子实验
    ap.add_argument("--sleep", type=float, default=0.5)
    a = ap.parse_args()

    tmpl = open(PROMPT_FILE[a.version], encoding="utf-8").read()
    samples = json.load(open(a.sample_file, encoding="utf-8"))
    if a.subset:
        keep = set(a.subset.split(","))
        samples = [s for s in samples if s["sample_id"] in keep]

    raw, timing = {}, {}
    for i, s in enumerate(samples, 1):
        prompt = tmpl.replace("{job_description}", s["input_text"])
        t0 = time.time()                       # ← 运行时间测量点
        try:
            text, tin, tout = call_llm(prompt, a.model, a.temperature)
        except Exception as e:
            text, tin, tout = "[[CALL_FAILED]] %s" % e, 0, 0
        t1 = time.time()
        raw[s["sample_id"]] = text
        timing[s["sample_id"]] = {"runtime_s": round(t1 - t0, 4),
                                  "prompt_tokens": tin, "completion_tokens": tout}
        print("[%d/%d] %s  %.2fs  in=%d out=%d" %
              (i, len(samples), s["sample_id"], t1 - t0, tin, tout), flush=True)
        time.sleep(a.sleep)

    os.makedirs("outputs", exist_ok=True)
    json.dump(raw, open("outputs/%s_%s_raw.json" % (a.run, a.version), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(timing, open("outputs/%s_%s_timing.json" % (a.run, a.version), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    rts = [v["runtime_s"] for v in timing.values()]
    print("\n平均运行时间 = %.3f s (N=%d)" % (sum(rts) / len(rts), len(rts)))


if __name__ == "__main__":
    main()
