import re

lines = message.split(b"\n")
kept = [
    line
    for line in lines
    if not re.match(
        br"(?i)^co-authored-by:\s*cursor(\s+agent)?\s*<cursoragent@cursor\.com>\s*$",
        line.strip(),
    )
]
out = b"\n".join(kept)
if message.endswith(b"\n") and not out.endswith(b"\n"):
    out += b"\n"
return out
