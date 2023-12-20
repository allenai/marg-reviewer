import re

# Adapted from jupyterlab css
COLOR_TABLE = {
    "black": {"hex": "3e424d", "ansi": "30"},
    "red": {"hex": "e75c58", "ansi": "31"},
    "green": {"hex": "00a050", "ansi": "32"},
    "yellow": {"hex": "ddbb33", "ansi": "33"},
    "blue": {"hex": "2090ff", "ansi": "34"},
    "magenta": {"hex": "d060c0", "ansi": "35"},
    "cyan": {"hex": "60c7c7", "ansi": "36"},
    "white": {"hex": "c0c0b0", "ansi": "37"},
    "strong-black": {"hex": "303030", "ansi": "90"},
    "strong-red": {"hex": "b03030", "ansi": "91"},
    "strong-green": {"hex": "007030", "ansi": "92"},
    "strong-yellow": {"hex": "b08010", "ansi": "93"},
    "strong-blue": {"hex": "0070dd", "ansi": "94"},
    "strong-magenta": {"hex": "a03090", "ansi": "95"},
    "strong-cyan": {"hex": "209090", "ansi": "96"},
    "strong-white": {"hex": "a0a0b0", "ansi": "97"},
}


def colorify(s: str, color: str, bold: bool = False, form="html", tag_side="both"):
    """if tag_side is 'left', only the left tag is added.  If tag_side irght
    'right', only the right tag is added.  This is useful if, for example,
    a list of tokens needs to be colored without joining the tokens.  Raises an
    error if this is not possible for the given form."""
    if color is None or form == "none":
        return s

    m = re.match(r"#(?P<hexcode>[0-9a-fA-F]{6})", color)
    valid_ansi = False
    if not m:
        if color in COLOR_TABLE:
            valid_ansi = True
            hex_color = COLOR_TABLE[color]["hex"]
        else:
            raise ValueError("Invalid color {}".format(color))
    else:
        hex_color = m.group("hexcode")

    left_tag, right_tag = "", ""
    if form == "html":
        bold_code = "font-weight: bold;" if bold else ""
        left_tag = '<span style="color: #{code};{boldness}">'.format(code=hex_color, boldness=bold_code)
        right_tag = "</span>"
    elif form == "ansi" and valid_ansi:
        bold_code = "1" if bold else "0"
        left_tag = "\033[{boldness};{code}m".format(code=COLOR_TABLE[color]["ansi"], boldness=bold_code)
        right_tag = "\033[0m"
    else:
        raise ValueError("Invalid format {}".format(form))

    if tag_side == "left":
        return left_tag + s
    elif tag_side == "right":
        return s + right_tag
    elif tag_side == "both":
        return left_tag + s + right_tag
    raise ValueError("Invalid tag_side {}".format(tag_side))


def colorprint(s, color=None, bold=False, form="ansi", *print_args, **print_kwargs):
    return print(colorify(s, color, bold=bold, form=form), *print_args, **print_kwargs)
