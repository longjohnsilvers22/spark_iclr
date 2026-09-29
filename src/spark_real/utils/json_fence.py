"""Pull the payload out of an LLM reply wrapped in a ``` fence."""


def strip_json_fence(txt: str) -> str:
    txt = txt.strip()
    if "```" in txt:
        txt = txt.split("```", 2)[1]
        if txt.startswith("json"):
            txt = txt[4:]
    return txt.strip()
