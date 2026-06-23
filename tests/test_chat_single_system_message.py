"""Regression test: the chat prompt must use exactly ONE system message.

Prod LLM "my-combo" (an OpenAI-compatible proxy) only honours the FIRST system
message. When the chain sent multiple system messages (main prompt + empty
extra_system + date guidance), the model ignored the retrieved context and
hallucinated the Ketua DPRD's name — even though the correct name was in the
context. Folding everything into one system message fixed it.

This test pins that contract so a future refactor can't re-introduce a second
system message.
"""

from langchain_core.messages import SystemMessage

from src.chat import build_qa_prompt


def _format(extra_system: str):
    prompt = build_qa_prompt()
    messages = prompt.format_messages(
        context="Ketua DPRD Tabalong, Riza Fahlipi, mengatakan ...",
        extra_system=extra_system,
        chat_history=[],
        input="siapa ketua dprd tabalong",
    )
    return messages


def test_exactly_one_system_message_when_extra_empty():
    msgs = _format(extra_system="")
    system_msgs = [m for m in msgs if isinstance(m, SystemMessage)]
    assert len(system_msgs) == 1, f"expected 1 system message, got {len(system_msgs)}"
    # context + date guidance must both survive inside that single message
    assert "Riza Fahlipi" in system_msgs[0].content
    assert "Tanggal hari ini" in system_msgs[0].content


def test_exactly_one_system_message_when_extra_present():
    msgs = _format(extra_system="Jawab singkat.")
    system_msgs = [m for m in msgs if isinstance(m, SystemMessage)]
    assert len(system_msgs) == 1, f"expected 1 system message, got {len(system_msgs)}"
    assert "Jawab singkat." in system_msgs[0].content
