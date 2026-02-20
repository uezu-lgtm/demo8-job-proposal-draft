from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

import requests


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


class LLMError(RuntimeError):
    pass


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return v


def _safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False)


def _extract_json(text: str) -> str:
    """
    Best-effort extraction of a JSON object from LLM output.
    Accepts raw JSON or fenced ```json blocks.
    """
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    if "```" in s:
        # Prefer ```json ... ```
        for fence in ("```json", "```JSON", "```"):
            if fence in s:
                parts = s.split(fence, 1)
                if len(parts) < 2:
                    continue
                after = parts[1]
                body = after.split("```", 1)[0]
                body = body.strip()
                if body.startswith("{") and body.endswith("}"):
                    return body
    # Fallback: try to find first { ... last }
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1].strip()
    return s


class BaseClient:
    name: str

    def complete(self, messages: list[ChatMessage], *, temperature: float = 0.2) -> str:
        raise NotImplementedError


class OllamaClient(BaseClient):
    def __init__(self, model: str, base_url: str):
        self.name = f"ollama:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete(self, messages: list[ChatMessage], *, temperature: float = 0.2) -> str:
        try:
            # 1) Native Ollama API
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "stream": False,
                "options": {"temperature": temperature},
            }
            r = requests.post(url, json=payload, timeout=120)
            if r.status_code == 404:
                # 2) OpenAI-compatible API (some setups expose only /v1/*)
                url_v1 = f"{self.base_url}/v1/chat/completions"
                payload_v1 = {
                    "model": self.model,
                    "messages": [{"role": m.role, "content": m.content} for m in messages],
                    "temperature": temperature,
                }
                r2 = requests.post(url_v1, json=payload_v1, timeout=120)
                r2.raise_for_status()
                ct2 = (r2.headers.get("Content-Type") or "").lower()
                if "application/json" not in ct2:
                    snippet2 = (r2.text or "")[:200].replace("\n", " ")
                    raise LLMError(
                        f"Ollama(v1互換)がJSONを返しませんでした（status={r2.status_code}, content-type={ct2}）。"
                        f" base_url={self.base_url} / snippet: {snippet2}"
                    )
                data2 = r2.json()
                return (((data2.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()

            r.raise_for_status()
            ct = (r.headers.get("Content-Type") or "").lower()
            if "application/json" not in ct:
                snippet = (r.text or "")[:200].replace("\n", " ")
                raise LLMError(
                    f"OllamaがJSONを返しませんでした（status={r.status_code}, content-type={ct}）。"
                    f" base_url={self.base_url} / snippet: {snippet}"
                )
            data = r.json()
            return (data.get("message", {}) or {}).get("content", "").strip()
        except Exception as e:  # pragma: no cover
            raise LLMError(f"Ollama 呼び出しに失敗しました: {e}") from e


class MockClient(BaseClient):
    def __init__(self):
        self.name = "mock"

    def complete(self, messages: list[ChatMessage], *, temperature: float = 0.2) -> str:
        # Deterministic “good enough” mock for demo when no LLM is available.
        _ = temperature
        joined = "\n".join([m.content for m in messages if m.role == "user"])
        # very rough heuristics
        job_lines = [ln.strip() for ln in joined.splitlines() if ln.strip()]
        job_hint = " / ".join(job_lines[:3])[:120]
        now = time.strftime("%Y-%m-%d")
        obj = {
            "metadata": {"generated_at": now, "match_score": 0.72, "llm": self.name},
            "evidence_points": [
                {
                    "title": "経験と業務内容の近さ",
                    "why": "過去の経験が求人の主要業務と重なり、立ち上がりが早い可能性があります。",
                    "evidence": [
                        {"source": "job", "quote": "（求人票）主要業務に関する記述が入力内にあります"},
                        {
                            "source": "candidate",
                            "quote": "（プロフィール）関連する経験・実績の記述が入力内にあります",
                        },
                    ],
                    "risk_or_gap": "具体的な担当範囲・KPIの期待値は要確認。",
                    "confirm_questions": ["直近で最も近い案件/役割での担当範囲を教えてください。"],
                },
                {
                    "title": "志向性・働き方条件の整合",
                    "why": "希望条件（働き方/成長/裁量）と求人の提供価値が近い可能性があります。",
                    "evidence": [
                        {"source": "job", "quote": "（求人票）働き方/魅力/制度の記述が入力内にあります"},
                        {"source": "candidate", "quote": "（プロフィール）希望条件の記述が入力内にあります"},
                    ],
                    "risk_or_gap": "出社頻度・給与レンジのすり合わせが必要。",
                    "confirm_questions": ["出社頻度の希望（許容範囲）を教えてください。"],
                },
            ],
            "proposal_short": "ご経験と志向に合いそうな求人をご紹介させてください。主要業務がこれまでのご経験と近く、立ち上がりが早い可能性があります。ご都合の良いお時間で10分ほどお話できますか？",
            "proposal_long": f"突然のご連絡失礼します。\n\n今回は、求人の主要業務がこれまでのご経験と近く、早期に価値発揮いただけそうだと感じたためご提案です。\n\n- マッチ理由（概要）：業務の近さ／志向性・働き方条件の整合\n- まず確認したい点：担当範囲、出社頻度、条件面\n\nもしご関心があれば、求人票の詳細と併せてポイントをご説明します。{now}週でご都合の良い候補日時を2〜3ついただけますでしょうか。",
            "checklist": [
                {
                    "category": "条件（Must）",
                    "items": [
                        {"text": "年収レンジ（下限/上限）の確認", "must": True},
                        {"text": "出社頻度/勤務地（リモート可否）の確認", "must": True},
                        {"text": "転職希望時期・稼働開始可能時期の確認", "must": True},
                    ],
                },
                {
                    "category": "経験・期待値",
                    "items": [
                        {"text": "主要業務に直結する経験の具体例を確認", "must": True},
                        {"text": "成果指標（KPI/役割期待）を求人側に確認", "must": False},
                    ],
                },
                {
                    "category": "注意点/NG",
                    "items": [
                        {"text": "入力にない情報を断定しない（不明は質問に回す）", "must": True}
                    ],
                },
            ],
            "confirm_questions": [
                "現年収/希望年収（下限）を差し支えない範囲で教えてください。",
                "出社頻度の希望（理想/許容）を教えてください。",
                "直近で最も近い経験の担当範囲・チーム規模を教えてください。",
            ],
            "debug": {"job_hint": job_hint},
        }
        return _safe_json_dumps(obj)


def build_client(provider: str) -> BaseClient:
    provider = provider.strip().lower()
    if provider == "ollama":
        model = _env("OLLAMA_MODEL", "llama3.1") or "llama3.1"
        base_url = _env("OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434"
        return OllamaClient(model=model, base_url=base_url)
    if provider == "mock":
        return MockClient()
    raise ValueError("provider は ollama / mock のいずれかです。")


def parse_llm_json(text: str) -> dict[str, Any]:
    raw = _extract_json(text)
    try:
        return json.loads(raw)
    except Exception as e:
        raise LLMError(f"JSONとして解析できませんでした。出力先頭200文字: {text[:200]}") from e

