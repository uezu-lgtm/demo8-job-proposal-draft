from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

import streamlit as st
from dotenv import load_dotenv
import requests

from llm_clients import ChatMessage, LLMError, build_client, parse_llm_json
from sample_data import (
    SAMPLE_CANDIDATE_1,
    SAMPLE_CANDIDATE_2,
    SAMPLE_JOB_1,
    SAMPLE_JOB_2,
    SAMPLE_PAST_PROPOSALS,
)


load_dotenv()


APP_TITLE = "求人提案オペ：下書き自動生成（求職者×求人）"


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _json_preview(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _inject_css() -> None:
    st.markdown(
        """
<style>
/* tighten page */
div.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }

.demo-hero {
  border: 1px solid rgba(15, 23, 42, 0.10);
  background: linear-gradient(180deg, rgba(37,99,235,0.08), rgba(255,255,255,0.0));
  border-radius: 16px;
  padding: 16px 16px;
  margin-bottom: 12px;
}
.demo-hero h3 { margin: 0 0 6px 0; font-weight: 700; }
.demo-hero p { margin: 0; color: rgba(15,23,42,0.75); }

.pill {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 999px;
  border: 1px solid rgba(15,23,42,0.12);
  background: rgba(255,255,255,0.9);
  font-size: 12px;
  margin-right: 6px;
}
.pill-strong {
  border-color: rgba(37,99,235,0.35);
  background: rgba(37,99,235,0.10);
}

.hint { color: rgba(15,23,42,0.7); font-size: 13px; }
</style>
        """,
        unsafe_allow_html=True,
    )


def _compact(text: str, limit: int = 60) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if len(t) <= limit:
        return t
    return t[: limit - 1] + "…"


def _as_markdown(obj: dict[str, Any]) -> str:
    meta = obj.get("metadata", {}) or {}
    lines: list[str] = []
    lines.append("# 求人提案 下書き")
    lines.append("")
    lines.append(f"- 生成: {meta.get('generated_at','-')}")
    if meta.get("provider"):
        lines.append(f"- LLM: {meta.get('provider')}")
    if meta.get("tone"):
        lines.append(f"- トーン: {meta.get('tone')}")
    if meta.get("output_detail"):
        lines.append(f"- 詳細度: {meta.get('output_detail')}")
    if meta.get("match_score") is not None:
        lines.append(f"- マッチ度（参考）: {meta.get('match_score')}")
    lines.append("")

    lines.append("## 提案文（短文）")
    lines.append("")
    lines.append(obj.get("proposal_short", "") or "")
    lines.append("")

    lines.append("## 提案文（長文）")
    lines.append("")
    lines.append(obj.get("proposal_long", "") or "")
    lines.append("")

    lines.append("## 根拠ポイント（引用つき）")
    lines.append("")
    for i, p in enumerate(obj.get("evidence_points", []) or [], start=1):
        lines.append(f"### {i}. {p.get('title','')}")
        lines.append("")
        lines.append(p.get("why", "") or "")
        lines.append("")
        lines.append("- 根拠（引用）")
        evs = p.get("evidence", []) or []
        if evs:
            for ev in evs:
                quote = (ev.get("quote") or "").strip()
                note = (ev.get("note") or "").strip()
                src = ev.get("source") or "-"
                lines.append(f"  - [{src}] {quote}" + (f"（{note}）" if note else ""))
        else:
            lines.append("  - （なし）")
        risk = (p.get("risk_or_gap") or "").strip()
        if risk:
            lines.append(f"- 懸念/ギャップ: {risk}")
        qs = p.get("confirm_questions", []) or []
        if qs:
            lines.append("- 確認質問")
            for q in qs:
                lines.append(f"  - {q}")
        lines.append("")

    lines.append("## 送付前チェックリスト")
    lines.append("")
    for c in obj.get("checklist", []) or []:
        lines.append(f"### {c.get('category','')}")
        items = c.get("items", []) or []
        for it in items:
            must = bool(it.get("must", False))
            lines.append(f"- [ ] {'Must' if must else 'Should'}: {it.get('text','')}")
        lines.append("")

    lines.append("## 確認質問（全体）")
    lines.append("")
    for q in obj.get("confirm_questions", []) or []:
        lines.append(f"- {q}")
    lines.append("")
    return "\n".join(lines)


def build_prompt(
    *,
    job_text: str,
    candidate_text: str,
    past_examples: str,
    tone: str,
    advisor_role_name: str,
    output_detail: str,
) -> list[ChatMessage]:
    system = f"""あなたは人材紹介の{advisor_role_name}です。目的は「求人提案の下書きを高速に、品質を揃えて作る」ことです。

出力ルール（最重要）:
- 日本語で出力
- 返答は **必ずJSONのみ**（前後に説明文・markdown・コードフェンス禁止）
- 入力に存在しない情報は断定しない。不明な場合は「確認質問」に回す
- 根拠ポイントには、必ず「引用（quote）」を添える。引用は原文からの抜き出しで、長くても60文字
- 引用が取れない場合は quote を空文字にし、note に「引用箇所不明」と書く

あなたの仕事:
1) マッチ理由（根拠）を抽出（良い点/懸念/要確認を分離）
2) 提案文（短文/長文）を生成（トーン: {tone}、詳細度: {output_detail}）
3) 送付時の注意点（条件/確認事項）をチェックリスト化（Must/Should）
4) 確認質問を具体的に提示（送付前・面談で聞く）

出力JSONスキーマ:
{{
  "metadata": {{
    "generated_at": "{_now_iso()}",
    "tone": "{tone}",
    "output_detail": "{output_detail}",
    "match_score": 0.0
  }},
  "evidence_points": [
    {{
      "title": "string",
      "why": "string",
      "evidence": [
        {{
          "source": "job|candidate|past",
          "quote": "string",
          "note": "string"
        }}
      ],
      "risk_or_gap": "string",
      "confirm_questions": ["string"],
      "confidence": "high|medium|low"
    }}
  ],
  "proposal_short": "string",
  "proposal_long": "string",
  "checklist": [
    {{
      "category": "string",
      "items": [
        {{
          "text": "string",
          "must": true
        }}
      ]
    }}
  ],
  "confirm_questions": ["string"]
}}
"""

    user = f"""以下の入力をもとに、スキーマに従ってJSONのみ出力してください。

【求人票】
{job_text}

【求職者プロフィール】
{candidate_text}

【過去提案例（スタイル参考。内容の事実は参照しない）】
{past_examples}
"""

    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]


def validate_result(obj: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    for k in ["metadata", "evidence_points", "proposal_short", "proposal_long", "checklist", "confirm_questions"]:
        if k not in obj:
            errs.append(f"キー欠落: {k}")
    if "evidence_points" in obj and not isinstance(obj["evidence_points"], list):
        errs.append("evidence_points は配列である必要があります")
    if "checklist" in obj and not isinstance(obj["checklist"], list):
        errs.append("checklist は配列である必要があります")
    if "confirm_questions" in obj and not isinstance(obj["confirm_questions"], list):
        errs.append("confirm_questions は配列である必要があります")
    return errs


def try_ollama_tags(base_url: str) -> tuple[bool, str]:
    try:
        base = base_url.rstrip("/")

        # 1) Native endpoint
        url = base + "/api/tags"
        r = requests.get(url, timeout=8)
        ct = (r.headers.get("Content-Type") or "").lower()
        if "application/json" not in ct:
            snippet = (r.text or "")[:120].replace("\n", " ")
            return (
                False,
                f"JSONではない応答です（status={r.status_code}, content-type={ct}）。"
                f" BASE_URLがOllamaでない可能性があります。例: http://localhost:11434 / snippet: {snippet}",
            )
        if r.status_code == 404:
            # 2) OpenAI-compatible endpoint
            url2 = base + "/v1/models"
            r2 = requests.get(url2, timeout=8)
            ct2 = (r2.headers.get("Content-Type") or "").lower()
            if "application/json" not in ct2:
                snippet2 = (r2.text or "")[:120].replace("\n", " ")
                return (
                    False,
                    f"JSONではない応答です（status={r2.status_code}, content-type={ct2}）。"
                    f" BASE_URLがOllamaでない可能性があります。例: http://localhost:11434 / snippet: {snippet2}",
                )
            r2.raise_for_status()
            data2 = r2.json()
            models2 = [m.get("id", "") for m in (data2.get("data") or [])]
            models2 = [m for m in models2 if m]
            if models2:
                return True, " / ".join(models2[:8]) + (" …" if len(models2) > 8 else "")
            return True, "（モデル一覧は空でした）"

        r.raise_for_status()
        data = r.json()
        models = [m.get("name", "") for m in (data.get("models") or [])]
        models = [m for m in models if m]
        if models:
            return True, " / ".join(models[:8]) + (" …" if len(models) > 8 else "")
        return True, "（モデル一覧は空でした）"
    except Exception as e:
        return False, str(e)


def repair_json_with_llm(client, raw: str, schema_hint: str) -> str:
    messages = [
        ChatMessage(
            role="system",
            content="あなたはJSON整形器です。与えられたテキストを、指示されたJSONスキーマに沿う「JSONのみ」に整形して返してください。余計な文章は禁止。",
        ),
        ChatMessage(
            role="user",
            content=f"この出力をJSONに整形してください。スキーマ要件: {schema_hint}\n\n---\n{raw}\n---",
        ),
    ]
    return client.complete(messages, temperature=0.0)


def render_result(obj: dict[str, Any]) -> None:
    meta = obj.get("metadata", {})
    top = st.columns([1, 1, 2], vertical_alignment="center")
    with top[0]:
        st.metric("マッチ度（参考）", f"{meta.get('match_score','-')}")
    with top[1]:
        st.metric("根拠ポイント数", f"{len(obj.get('evidence_points', []) or [])}")
    with top[2]:
        st.caption(
            f"生成: {meta.get('generated_at', '-') } / LLM={meta.get('provider','-')} / tone={meta.get('tone','-')} / detail={meta.get('output_detail','-')}"
        )

    tab1, tab2, tab3, tab4 = st.tabs(["根拠ポイント", "提案文（短文）", "提案文（長文）", "チェックリスト/確認質問"])

    with tab1:
        pts = obj.get("evidence_points", []) or []
        if not pts:
            st.info("根拠ポイントがありません。入力を増やすか、過去提案例を入れて再生成してください。")
        for i, p in enumerate(pts, start=1):
            title = p.get("title", f"ポイント{i}")
            conf = p.get("confidence", "-")
            with st.expander(f"{i}. {title}  / confidence: {conf}", expanded=i <= 2):
                st.markdown("**なぜマッチするか**")
                st.write(p.get("why", ""))
                st.markdown("**根拠（引用）**")
                evs = p.get("evidence", []) or []
                if evs:
                    for ev in evs:
                        st.write(f"- **{ev.get('source','-')}**: 「{_compact(ev.get('quote',''))}」 {('— ' + ev.get('note','')) if ev.get('note') else ''}")
                else:
                    st.write("- （なし）")
                st.markdown("**懸念/ギャップ**")
                st.write(p.get("risk_or_gap", ""))
                st.markdown("**このポイントに関する確認質問**")
                qs = p.get("confirm_questions", []) or []
                if qs:
                    for q in qs:
                        st.write(f"- {q}")
                else:
                    st.write("- （なし）")

    with tab2:
        st.text_area("短文（コピペ用）", value=obj.get("proposal_short", ""), height=180)

    with tab3:
        st.text_area("長文（送付文案）", value=obj.get("proposal_long", ""), height=360)

    with tab4:
        st.markdown("**送付前チェックリスト**")
        checklist = obj.get("checklist", []) or []
        if not checklist:
            st.write("（なし）")
        for c in checklist:
            cat = c.get("category", "カテゴリ")
            st.markdown(f"**{cat}**")
            items = c.get("items", []) or []
            for it in items:
                must = bool(it.get("must", False))
                label = f"{'Must' if must else 'Should'}: {it.get('text','')}"
                st.checkbox(label, value=False, key=f"chk::{cat}::{label}")

        st.divider()
        st.markdown("**確認質問（全体）**")
        for q in obj.get("confirm_questions", []) or []:
            st.write(f"- {q}")

        with st.expander("JSON（デバッグ）"):
            st.code(_json_preview(obj))


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📝")
    _inject_css()
    st.title(APP_TITLE)
    st.markdown(
        """
<div class="demo-hero">
  <h3>“提案が一瞬で出る”体験</h3>
  <p>求人票×求職者プロフィールから、提案文案（短文/長文）・根拠（引用）・送付前チェックリストを自動生成します。</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("設定")
        provider = st.selectbox("LLMプロバイダ", ["ollama", "mock"], index=0)
        tone = st.selectbox("トーン", ["丁寧", "フランク（丁寧寄り）", "硬め（ビジネス）"], index=0)
        output_detail = st.selectbox("詳細度", ["短め", "標準", "丁寧め"], index=1)
        advisor_role_name = st.text_input("役割呼称（表示/プロンプト用）", value="キャリアアドバイザー")
        temperature = st.slider("温度（ブレ）", 0.0, 1.0, 0.2, 0.05)

        st.divider()
        st.caption("Ollama（任意）")
        st.session_state.setdefault("ollama_base_url", os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
        st.session_state.setdefault("ollama_model", os.environ.get("OLLAMA_MODEL", "llama3.1"))

        reset_cols = st.columns([1, 1])
        if reset_cols[0].button("11434に戻す", use_container_width=True):
            st.session_state["ollama_base_url"] = "http://localhost:11434"
        if reset_cols[1].button("アプリURLを入れてしまった", use_container_width=True):
            st.session_state["ollama_base_url"] = "http://localhost:11434"

        base_url = st.text_input(
            "OLLAMA_BASE_URL",
            key="ollama_base_url",
            help="OllamaのURL（通常: http://localhost:11434）。※アプリのURL（例: http://localhost:8502）ではありません。",
        )
        model = st.text_input("OLLAMA_MODEL", key="ollama_model")
        if ":8502" in base_url or base_url.rstrip("/").endswith(":8502"):
            st.warning("OLLAMA_BASE_URL がアプリ(8502)を指しています。通常は http://localhost:11434 です。")
        if st.button("Ollama接続チェック"):
            ok, msg = try_ollama_tags(base_url)
            if ok:
                st.success(f"接続OK。models: {msg}")
            else:
                st.error(f"接続NG: {msg}")

        st.divider()
        st.caption("環境変数（メモ）")
        st.code(
            "OLLAMA_BASE_URL / OLLAMA_MODEL",
            language="text",
        )

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.subheader("入力")
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("サンプル投入（1）"):
            st.session_state["job_text"] = SAMPLE_JOB_1
            st.session_state["candidate_text"] = SAMPLE_CANDIDATE_1
            st.session_state["past_examples"] = SAMPLE_PAST_PROPOSALS
        if b2.button("サンプル投入（2）"):
            st.session_state["job_text"] = SAMPLE_JOB_2
            st.session_state["candidate_text"] = SAMPLE_CANDIDATE_2
            st.session_state["past_examples"] = SAMPLE_PAST_PROPOSALS
        if b3.button("過去提案例だけ投入"):
            st.session_state["past_examples"] = SAMPLE_PAST_PROPOSALS
        if b4.button("クリア"):
            st.session_state["job_text"] = ""
            st.session_state["candidate_text"] = ""
            st.session_state["past_examples"] = ""
            st.session_state.pop("last_raw", None)
            st.session_state.pop("last_obj", None)

        st.markdown(
            f"""<span class="pill pill-strong">入力</span>
<span class="pill">求人票</span><span class="pill">プロフィール</span><span class="pill">過去提案例(任意)</span>""",
            unsafe_allow_html=True,
        )
        job_text = st.text_area("求人票テキスト", key="job_text", height=250, placeholder="ここに求人票のテキストを貼り付け")
        candidate_text = st.text_area(
            "求職者プロフィール",
            key="candidate_text",
            height=250,
            placeholder="ここに求職者プロフィールのテキストを貼り付け",
        )
        past_examples = st.text_area(
            "過去提案例（任意・スタイル参考）",
            key="past_examples",
            height=160,
            placeholder="良い提案の例（匿名）を貼り付け。無くても動きます。",
        )

        disabled = not (job_text.strip() and candidate_text.strip())
        generate = st.button("下書きを生成", type="primary", disabled=disabled, use_container_width=True)

        if disabled:
            st.info("求人票と求職者プロフィールを入れると生成できます。サンプル投入も使えます。")

    with col_out:
        st.subheader("出力")
        if generate:
            try:
                # apply sidebar inputs to env for this session
                os.environ["OLLAMA_BASE_URL"] = base_url
                os.environ["OLLAMA_MODEL"] = model
                client = build_client(provider)
                st.session_state["last_provider_name"] = getattr(client, "name", provider)
                messages = build_prompt(
                    job_text=job_text,
                    candidate_text=candidate_text,
                    past_examples=past_examples,
                    tone=tone,
                    advisor_role_name=advisor_role_name,
                    output_detail=output_detail,
                )
                with st.spinner(f"生成中…（{st.session_state['last_provider_name']}）"):
                    raw = client.complete(messages, temperature=float(temperature))
                st.session_state["last_raw"] = raw
                schema_hint = "metadata/evidence_points/proposal_short/proposal_long/checklist/confirm_questions を含む"
                try:
                    obj = parse_llm_json(raw)
                except LLMError:
                    # Best-effort repair once (Ollama output sometimes contains extra text)
                    repaired = repair_json_with_llm(client, raw, schema_hint)
                    st.session_state["last_raw"] = repaired
                    obj = parse_llm_json(repaired)
                # add some metadata we know
                obj.setdefault("metadata", {})
                obj["metadata"].setdefault("generated_at", _now_iso())
                obj["metadata"]["provider"] = st.session_state["last_provider_name"]
                errs = validate_result(obj)
                if errs:
                    st.error("JSONは読めましたが、期待スキーマと差分があります。")
                    for e in errs:
                        st.write(f"- {e}")
                    with st.expander("生出力"):
                        st.code(raw)
                    with st.expander("JSON（解析後）"):
                        st.code(_json_preview(obj))
                else:
                    st.session_state["last_obj"] = obj
                    render_result(obj)

                    md = _as_markdown(obj)
                    st.divider()
                    dl = st.columns([1, 1, 2])
                    with dl[0]:
                        st.download_button(
                            "提案一式（Markdown）をDL",
                            data=md.encode("utf-8"),
                            file_name="proposal_draft.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )
                    with dl[1]:
                        st.download_button(
                            "JSONをDL",
                            data=_json_preview(obj).encode("utf-8"),
                            file_name="proposal_result.json",
                            mime="application/json",
                            use_container_width=True,
                        )
                    with dl[2]:
                        st.caption("DLしたMarkdownをそのまま社内共有・レビューに回せます。")
            except LLMError as e:
                st.error(str(e))
                raw = st.session_state.get("last_raw")
                if raw:
                    with st.expander("直近の生出力"):
                        st.code(raw)
            except Exception as e:  # pragma: no cover
                st.error(f"予期しないエラー: {e}")
                raw = st.session_state.get("last_raw")
                if raw:
                    with st.expander("直近の生出力"):
                        st.code(raw)
        else:
            if "last_obj" in st.session_state:
                render_result(st.session_state["last_obj"])
            else:
                st.info("まだ生成していません。左で入力して「下書きを生成」を押してください。")

    st.divider()
    with st.expander("運用上の注意（デモ設計）"):
        st.write(
            "- 出力は下書きです。送付前にチェックリストと根拠引用を必ず確認してください。\n"
            "- 入力にない情報は断定せず、確認質問として提示する設計です。\n"
            "- 過去提案例は“文体/構成”の参考としてのみ使用し、事実情報は参照しません。"
        )


if __name__ == "__main__":
    main()

