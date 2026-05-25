# Ex.No.6 Development of Python Code Compatible with Multiple AI Tools
# Aim: 
Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights with Multiple AI Tools

# AI Tools Required:
OpenAI  
Anthropic  
Google Gemini  
OpenRouter  
AIMLAPI  
LangChain  
LangGraph  
Semantic Kernel  
LLM Comparator  
PAIR llm-comparator

# Explanation:
Experiment the persona pattern as a programmer for any specific applications related with your interesting area. 
Generate the outoput using more than one AI tool and based on the code generation analyse and discussing that. 
# PROMPT: 
GIVE ME A PYTHON CODE TO IMPLEMENT A MODEL THAT INTEGRATES WITH MULTIPLE AI TOOLS TO AUTOMATE THE TASK OF INTERACTING WITH APIS, COMPARING OUTPUTS, AND GENERATING ACTIONABLE INSIGHTS WITH MULTIPLE AI TOOLS.

# OUTPUT:
Here is a minimal but complete Python pattern you can adapt to call multiple AI APIs in parallel, compare their outputs, and turn them into actionable insights.[1][2]

## Overview

The pattern below:

- Wraps each AI provider in a common interface.
- Uses `asyncio` to call all models in parallel.
- Normalizes and compares responses to extract concise **insights**.

You can plug in any APIs (OpenAI, Anthropic, Gemini, local models, etc.) by filling in the placeholders.

***

## 1. Setup and Model Abstractions

```python
import os
import asyncio
import json
from dataclasses import dataclass
from typing import Dict, Any, List

import aiohttp  # pip install aiohttp

# Load keys from env or config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_KEY")


@dataclass
class ModelResponse:
    provider: str
    model: str
    text: str
    usage: Dict[str, Any]


class BaseModelClient:
    provider: str

    async def generate(self, session: aiohttp.ClientSession, prompt: str) -> ModelResponse:
        raise NotImplementedError
```

***

## 2. Example Clients for Different APIs

### OpenAI-style client

Adapt the URL/body to the current endpoint you use.

```python
class OpenAIClient(BaseModelClient):
    provider = "openai"

    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model

    async def generate(self, session: aiohttp.ClientSession, prompt: str) -> ModelResponse:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a concise, analytical assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        }

        async with session.post(url, headers=headers, json=payload, timeout=60) as resp:
            data = await resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return ModelResponse(provider=self.provider, model=self.model, text=text, usage=usage)
```

### Anthropic-style client

```python
class AnthropicClient(BaseModelClient):
    provider = "anthropic"

    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        self.model = model

    async def generate(self, session: aiohttp.ClientSession, prompt: str) -> ModelResponse:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 512,
            "temperature": 0.3,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "system": "You are a concise, analytical assistant.",
        }

        async with session.post(url, headers=headers, json=payload, timeout=60) as resp:
            data = await resp.json()
            text = "".join(block["text"] for block in data["content"] if block["type"] == "text")
            usage = data.get("usage", {})
            return ModelResponse(provider=self.provider, model=self.model, text=text, usage=usage)
```

### Google Gemini-style client

```python
class GeminiClient(BaseModelClient):
    provider = "gemini"

    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = model

    async def generate(self, session: aiohttp.ClientSession, prompt: str) -> ModelResponse:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 512},
        }

        async with session.post(url, headers=headers, json=payload, timeout=60) as resp:
            data = await resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata", {})
            return ModelResponse(provider=self.provider, model=self.model, text=text, usage=usage)
```

***

## 3. Orchestrator: Call All Models in Parallel

```python
class MultiAIEvaluator:
    def __init__(self, clients: List[BaseModelClient]):
        self.clients = clients

    async def _run_single(self, session: aiohttp.ClientSession, client: BaseModelClient, prompt: str):
        try:
            return await client.generate(session, prompt)
        except Exception as e:
            return ModelResponse(
                provider=client.provider,
                model=getattr(client, "model", "unknown"),
                text=f"ERROR: {e}",
                usage={},
            )

    async def run_all(self, prompt: str) -> List[ModelResponse]:
        async with aiohttp.ClientSession() as session:
            tasks = [self._run_single(session, c, prompt) for c in self.clients]
            return await asyncio.gather(*tasks)
```

This pattern (async tasks over a list of tools) is standard for orchestrating multiple agents or models.[6][3]

***

## 4. Simple Automatic Comparison & Insight Generation

Here, a “judge” prompt is used on one of your LLMs to produce structured comparison and **actionable** summary. This mirrors side‑by‑side evaluation patterns used in LLM comparator tools.

```python
async def auto_compare_and_summarize(
    judge_client: BaseModelClient,
    prompt: str,
    model_outputs: List[ModelResponse],
) -> Dict[str, Any]:
    """Ask one model to act as a judge over the others."""

    comparison_payload = {
        "user_prompt": prompt,
        "responses": [
            {
                "provider": r.provider,
                "model": r.model,
                "text": r.text,
            }
            for r in model_outputs
        ],
    }

    judge_prompt = (
        "You are an expert evaluator comparing multiple AI model outputs.\n\n"
        "Given the original user prompt and the model responses (in JSON), do the following:\n"
        "1. Briefly describe the main points of agreement and disagreement.\n"
        "2. Identify which response is most reliable and why.\n"
        "3. List concrete, actionable recommendations for a user based on the BEST response.\n"
        "4. Output a JSON object with keys: 'best_model', 'reasoning', 'agreements', "
        "'disagreements', 'action_items'.\n\n"
        f"DATA:\n{json.dumps(comparison_payload, indent=2)}"
    )

    async with aiohttp.ClientSession() as session:
        judge_raw = await judge_client.generate(session, judge_prompt)
    # Very naive JSON extraction (you can improve with regex or structured-output tools). [web:18]
    text = judge_raw.text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    # Fallback: wrap in a generic container
    return {
        "best_model": None,
        "reasoning": text,
        "agreements": [],
        "disagreements": [],
        "action_items": [],
    }
```

Using one model as a judge for others is a common pattern in evaluation frameworks and research.
***

## 5. Putting It All Together

```python
async def main():
    user_prompt = (
        "You are helping an engineering student design a study plan for learning "
        "quantum computing and Python over 3 months. Provide detailed, practical steps."
    )

    clients = [
        OpenAIClient("gpt-4.1-mini"),
        AnthropicClient("claude-3-5-sonnet-20241022"),
        GeminiClient("gemini-1.5-flash"),
    ]

    orchestrator = MultiAIEvaluator(clients)
    model_outputs = await orchestrator.run_all(user_prompt)

    print("=== Raw Model Outputs ===")
    for r in model_outputs:
        print(f"\n[{r.provider}:{r.model}]")
        print(r.text[:800], "...\n")

    # Use one of the models (e.g., OpenAI) as the judge
    judge_client = OpenAIClient("gpt-4.1-mini")
    insights = await auto_compare_and_summarize(
        judge_client=judge_client,
        prompt=user_prompt,
        model_outputs=model_outputs,
    )

    print("\n=== Aggregated Insights ===")
    print(json.dumps(insights, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
```

This main flow:

- Sends the same task to multiple tools/models.
- Prints side‑by‑side outputs for manual inspection.
- Automatically derives a compact JSON of **best model** plus action items

***

## 6. How to Extend This

- Plug in other tools (web search, vector DB, internal APIs) by wrapping them in `BaseModelClient` or sibling abstractions.
- Add logging of latency, cost, and token usage to pick the best cost–performance trade‑off.
- Persist `model_outputs` and `insights` to a database or dashboard for continuous evaluation.

## OUTPUT COMPARISION: 
PROMPT: COMPARE THE OUTPUT FROM DIFFERENT APIs AND HIGHLIGHT THE DIFFERENCE.<BR>
**OUTPUT:**
OpenAI GPT, Anthropic Claude, and Google Gemini produce distinct outputs on the same prompts.[1][2]

### Key Differences

| Aspect | OpenAI GPT | Anthropic Claude | Google Gemini |
|--------|------------|------------------|---------------|
| **Style** | Creative, conversational, human-like flair [2][3] | Guarded, thorough, safety-focused explanations [2][3] | Factual, consistent, data-rich [1][2] |
| **Strengths** | Creativity, flexibility, multimodal tasks [1][4] | Reasoning, large inputs, fewer hallucinations [2][3] | Speed, accuracy, long-context [2][5] |
| **Speed** | GPT-4o: 2x faster TTFT (0.56s), 56 tokens/s [6] | Slower (1.23s TTFT), 28 tokens/s [6] | Fastest overall, <300ms latency [5] |
| **Accuracy** | Higher on facts (40% improvement), but creative risks [2][6] | Correct on technical details [6] | Top factual consistency [2][3] |

GPT leads creativity/accuracy, Claude excels reasoning/safety, Gemini wins speed/facts.[5][4][1]

## INSIGHTS: 
PROMPT: GIVE ME THE MEANINGFUL INSIGHTS FOR THESE
**OUTPUT:**
**For multi-AI orchestration, pick models by task type, not one-size-fits-all.**

### Actionable Insights

- **Creative tasks** (writing, ideation): Use GPT-4o/4.1 first[1][2]
- **Technical/reasoning** (code, analysis): Claude 3.5 Sonnet leads[3][4]
- **Speed/facts** (research, Q&A): Gemini 1.5 Flash/Pro[5][3]

**Orchestration strategy**: Send prompts to **all three in parallel**, let a "judge" model (Claude) score/rank them automatically. This gets best-of-breed results every time.[6][1]

**Cost optimization**: Gemini fastest/lowest latency, GPT highest quality/creativity value.[4]

# Conclusion:
Use OpenRouter or AIMLAPI as your single aggregator for multiple LLMs, pair it with asyncio in Python for parallel calls, and add a simple "judge" prompt for automated comparisons and actionable insights. This minimal stack delivers robust multi-AI orchestration without complexity.



# Result: 
The corresponding Prompt is executed successfully.
