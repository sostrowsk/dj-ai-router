import bleach
import markdown

ALLOWED_TAGS = [
    "p",
    "br",
    "strong",
    "em",
    "u",
    "del",
    "code",
    "pre",
    "a",
    "ul",
    "ol",
    "li",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
    "sup",
    "sub",
    "span",
    "div",
]
ALLOWED_ATTRIBUTES = {
    "a": ["href", "title"],
    "code": ["class"],
    "span": ["class"],
    "div": ["class"],
}


def safe_markdown_to_html(text, extensions=None):
    if not text:
        return ""
    if extensions is None:
        extensions = []
    html = markdown.markdown(text, extensions=extensions, output_format="html5")
    return bleach.clean(html, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRIBUTES, strip=True)
