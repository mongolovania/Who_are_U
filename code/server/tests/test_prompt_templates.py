"""5 阶段对话提示词模板测试"""

import pytest
from app.services.prompt_templates import (
    Stage,
    STAGE_MAP,
    STAGE_FLOW,
    get_stage_prompt,
    get_next_stage,
    extract_transition_signal,
    is_conversation_end,
)


def test_stage_flow_count():
    """测试共 5 个阶段"""
    assert len(STAGE_FLOW) == 5


def test_stage_flow_order():
    """测试阶段顺序正确"""
    assert STAGE_FLOW == [
        Stage.EMPATHY,
        Stage.SELF_SCAN,
        Stage.OPTION_BREAKDOWN,
        Stage.INNER_CONFIRM,
        Stage.DECISION_OUTPUT,
    ]


def test_all_stages_have_prompts():
    """测试每个阶段都有对应的提示词"""
    for stage in Stage:
        assert stage in STAGE_MAP, f"Missing prompt for {stage}"
        prompt = STAGE_MAP[stage]
        assert len(prompt.system_prompt) > 100, f"Prompt too short for {stage}"
        assert prompt.max_rounds > 0


def test_stage_name_cn():
    """测试每个阶段都有中文名称"""
    for stage in Stage:
        prompt = STAGE_MAP[stage]
        assert prompt.name_cn, f"Missing Chinese name for {stage}"
        assert len(prompt.name_cn) > 1


def test_get_stage_prompt():
    """测试获取阶段提示词"""
    prompt = get_stage_prompt(Stage.EMPATHY)
    assert prompt.stage == Stage.EMPATHY
    assert "共情" in prompt.system_prompt


def test_get_next_stage():
    """测试阶段流转"""
    assert get_next_stage(Stage.EMPATHY) == Stage.SELF_SCAN
    assert get_next_stage(Stage.SELF_SCAN) == Stage.OPTION_BREAKDOWN
    assert get_next_stage(Stage.OPTION_BREAKDOWN) == Stage.INNER_CONFIRM
    assert get_next_stage(Stage.INNER_CONFIRM) == Stage.DECISION_OUTPUT
    assert get_next_stage(Stage.DECISION_OUTPUT) is None


def test_extract_transition_signal():
    """测试过渡信号提取"""
    text = "... [TRANSITION:self_scan]"
    assert extract_transition_signal(text) == Stage.SELF_SCAN

    text = "你做得很好。[TRANSITION:option_breakdown]"
    assert extract_transition_signal(text) == Stage.OPTION_BREAKDOWN

    # 无信号
    assert extract_transition_signal("普通的回复") is None

    # 无效信号
    assert extract_transition_signal("[TRANSITION:invalid_stage]") is None


def test_is_conversation_end():
    """测试对话结束信号"""
    assert is_conversation_end("[END] 祝你一切顺利")
    assert not is_conversation_end("还没有结束")
    assert not is_conversation_end("[TRANSITION:decision_output]")


def test_prompt_templates_not_empty():
    """测试所有阶段提示词都不为空"""
    for stage in Stage:
        prompt = STAGE_MAP[stage].system_prompt
        assert len(prompt) > 50, f"Prompt for {stage} is too short: {len(prompt)} chars"


def test_transition_signals_consistent():
    """测试过渡信号名称与实际 Stage 枚举一致"""
    for stage in Stage:
        prompt = STAGE_MAP[stage]
        if stage != Stage.DECISION_OUTPUT:
            assert prompt.transition_signal, f"Missing transition_signal for {stage}"

    # 最后一个阶段用 [END]
    assert STAGE_MAP[Stage.DECISION_OUTPUT].transition_signal == "[END]"
