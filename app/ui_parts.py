"""Small HTML helpers for the MetaPilot UI (styles live in app/theme.css)."""
from html import escape

def hero(title="MetaPilot", tagline="Upload a table. Get the right model to try first."):
    return f'<div class="mp-hero"><span class="mp-ribbon">{escape(title)}</span><p class="mp-tag">{escape(tagline)}</p></div>'

def step(n, title, hint=""):
    h = f'<span class="h">{escape(hint)}</span>' if hint else ""
    return f'<div class="mp-step"><span class="n">{n}</span><span class="t">{escape(title)}</span>{h}</div>'

def ticket(rank, name, family="", bullets=(), label="ADMIT ONE", small=False, alt=False):
    cls = "mp-ticket" + (" small" if small else "") + (" alt" if alt else "")
    fam = f'<span class="mp-fam">{escape(family)}</span>' if family else ""
    why = "".join(f"<li>{escape(b)}</li>" for b in bullets)
    why = f'<ul class="mp-why">{why}</ul>' if why else ""
    return (f'<div class="{cls}"><div class="mp-body"><div class="mp-title">{escape(name)}</div>{fam}{why}</div>'
            f'<div class="mp-stub"><span class="no">No.{rank}</span><span class="lbl">{escape(label)}</span></div></div>')

def tickets(*items):
    return '<div class="mp-tickets">' + "".join(items) + "</div>"

def note(text_html):
    """text_html may contain <b>; escape any user text before passing it in."""
    return f'<div class="mp-note">{text_html}</div>'

def chips(items):
    return "".join(f'<span class="mp-chip">{escape(str(i))}</span>' for i in items)
