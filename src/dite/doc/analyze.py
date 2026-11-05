"""语义分析模块 - Single-Call Analyzer"""

import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from dite.app.config import Config
from dite.app.i18n import get_locale, t
from dite.util.llm import strip_reasoning_text
from dite.util.log import get_logger


@dataclass
class LayoutInfo:
    """布局信息"""

    type: str = "其他"  # 学术论文|技术报告|发票|合同|讲义|演示文稿|图片|其他
    columns: str = "single"  # single|double|triple|mixed
    has_table: bool = False
    has_logo: bool = False
    logo_position: str = "none"  # top-left|top-right|center|bottom|none
    visual_elements: list[str] = field(
        default_factory=list
    )  # chart, formula, code, image


@dataclass
class ContentInfo:
    """内容信息"""

    topic: str = ""  # 一个词概括主题
    keywords: list[str] = field(default_factory=list)  # 关键词列表
    entities: list[str] = field(default_factory=list)  # 公司名/人名/产品名
    language: str = "zh"  # zh|en|mixed
    domain: str = "other"  # tech|finance|legal|education|medical|other


@dataclass
class DocumentAnalysis:
    """文档分析结果"""

    layout: LayoutInfo = field(default_factory=LayoutInfo)
    content: ContentInfo = field(default_factory=ContentInfo)
    summary: str = ""  # 一句话摘要
    template_hints: str = ""  # 可能来自的模板或机构
    confidence: float = 0.0  # 置信度 0.0-1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentAnalysis":
        """从字典创建 DocumentAnalysis"""
        layout_data = data.get("layout", {})
        content_data = data.get("content", {})

        layout = LayoutInfo(
            type=layout_data.get("type", "其他"),
            columns=layout_data.get("columns", "single"),
            has_table=layout_data.get("has_table", False),
            has_logo=layout_data.get("has_logo", False),
            logo_position=layout_data.get("logo_position", "none"),
            visual_elements=layout_data.get("visual_elements", []),
        )

        content = ContentInfo(
            topic=content_data.get("topic", ""),
            keywords=content_data.get("keywords", []),
            entities=content_data.get("entities", []),
            language=content_data.get("language", "zh"),
            domain=content_data.get("domain", "other"),
        )

        return cls(
            layout=layout,
            content=content,
            summary=data.get("summary", ""),
            template_hints=data.get("template_hints", ""),
            confidence=data.get("confidence", 0.0),
        )


def _analysis_truncation_notice() -> str:
    """Return a locale-aware truncation marker for prompts."""
    if get_locale() == "zh-CN":
        return "\n\n[内容已截断...]"
    return "\n\n[Content truncated...]"


def _analysis_prompt(content: str) -> str:
    """Build a locale-aware analysis prompt."""
    if get_locale() == "zh-CN":
        return f"""分析以下文档，提取结构化信息。

**要求**：输出严格的 JSON 格式，不要包含任何解释性文字。

```json
{{
  "layout": {{
    "type": "学术论文|技术报告|发票|合同|讲义|演示文稿|图片|其他",
    "columns": "single|double|triple|mixed",
    "has_table": boolean,
    "has_logo": boolean,
    "logo_position": "top-left|top-right|center|bottom|none",
    "visual_elements": ["chart", "formula", "code", "image"]
  }},
  "content": {{
    "topic": "一个词概括主题",
    "keywords": ["关键词1", "关键词2", "关键词3"],
    "entities": ["公司名/人名/产品名"],
    "language": "zh|en|mixed",
    "domain": "tech|finance|legal|education|medical|other"
  }},
  "summary": "一句话摘要（不超过50字）",
  "template_hints": "可能来自的模板或机构（如：arXiv论文、某银行账单）",
  "confidence": 0.0-1.0
}}
```

**重要**：
- 必须以 `{{` 开头，以 `}}` 结尾
- 不要输出 Markdown 代码块
- 所有字符串使用双引号

文档内容：
{content}
"""

    return f"""Analyze the following document and extract structured information.

Requirements: output strict JSON only, with no explanatory text.

```json
{{
  "layout": {{
    "type": (
      "academic paper|technical report|invoice|contract|handout|"
      "presentation|image|other"
    ),
    "columns": "single|double|triple|mixed",
    "has_table": boolean,
    "has_logo": boolean,
    "logo_position": "top-left|top-right|center|bottom|none",
    "visual_elements": ["chart", "formula", "code", "image"]
  }},
  "content": {{
    "topic": "one-word topic",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "entities": ["company/person/product"],
    "language": "zh|en|mixed",
    "domain": "tech|finance|legal|education|medical|other"
  }},
  "summary": "one-sentence summary (max 50 words)",
  "template_hints": (
    "possible source template or organization "
    "(for example: arXiv paper, bank statement)"
  ),
  "confidence": 0.0-1.0
}}
```

Important:
- Must start with `{{` and end with `}}`
- Do not output Markdown code fences
- Use double quotes for all strings

Document content:
{content}
"""


def _build_weighted_payload_text(analysis: DocumentAnalysis, raw_content: str) -> str:
    """Build a locale-aware weighted embedding payload."""
    keywords = " ".join(analysis.content.keywords)
    excerpt = raw_content

    if get_locale() == "zh-CN":
        payload = f"""文档类型: {analysis.layout.type}
布局特征: {analysis.layout.columns}栏, 表格:{analysis.layout.has_table}
主题: {analysis.content.topic}
关键词: {keywords} {keywords} {keywords}
摘要: {analysis.summary}
内容节选: {excerpt}
"""
    else:
        payload = f"""Document type: {analysis.layout.type}
Layout: {analysis.layout.columns} columns, table:{analysis.layout.has_table}
Topic: {analysis.content.topic}
Keywords: {keywords} {keywords} {keywords}
Summary: {analysis.summary}
Excerpt: {excerpt}
"""
    return payload.strip()


def analyze_document(
    client: OpenAI,
    content: str,
    *,
    config: Config,
    max_content_length: int = 4000,
    max_retries: int = 3,
    llm_model: str | None = None,
) -> DocumentAnalysis:
    """
    使用 Single-Call 分析文档

    Args:
        client: OpenAI 兼容客户端
        content: 文档内容
        max_content_length: 最大内容长度（避免超出 token 限制）
        max_retries: 最大重试次数
        llm_model: 可选模型名（显式依赖注入优先）
        config: 可选配置对象

    Returns:
        DocumentAnalysis 对象
    """
    model = llm_model or config.models.llm
    logger = get_logger()

    # 截断内容
    if len(content) > max_content_length:
        content = content[:max_content_length] + _analysis_truncation_notice()

    prompt = _analysis_prompt(content)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},  # 强制 JSON 输出
                max_tokens=500,
            )

            response_text = (
                strip_reasoning_text(response.choices[0].message.content) or "{}"
            )

            # 尝试解析 JSON
            data = json.loads(response_text)
            return DocumentAnalysis.from_dict(data)

        except json.JSONDecodeError as e:
            logger.debug(
                t(
                    "debug_analyzer_json_parse_failed",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=e,
                )
            )
            if attempt == max_retries - 1:
                logger.warning(t("warning_analyzer_default_used"))
                return DocumentAnalysis()

        except Exception as e:
            logger.debug(
                t(
                    "debug_analyzer_api_failed",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=e,
                )
            )
            if attempt == max_retries - 1:
                logger.warning(t("warning_analyzer_failed", error=e))
                return DocumentAnalysis()

    return DocumentAnalysis()


def build_weighted_payload(
    analysis: DocumentAnalysis,
    raw_content: str,
    max_excerpt_length: int = 1000,
) -> str:
    """
    构建带权重的 Embedding 输入

    关键词重复 3 次以增加在 Embedding 中的权重

    Args:
        analysis: 文档分析结果
        raw_content: 原始内容
        max_excerpt_length: 内容节选的最大长度

    Returns:
        加权后的 payload 文本
    """
    # 关键词重复 3 次以增加权重
    return _build_weighted_payload_text(
        analysis,
        raw_content[:max_excerpt_length],
    )


def analyze_and_build_payload(
    client: OpenAI,
    content: str,
    *,
    config: Config,
) -> tuple[DocumentAnalysis, str]:
    """
    分析文档并构建加权 payload

    Args:
        client: OpenAI 兼容客户端
        content: 文档内容

    Returns:
        (DocumentAnalysis, weighted_payload)
    """
    analysis = analyze_document(client, content, config=config)
    payload = build_weighted_payload(analysis, content)
    return analysis, payload
