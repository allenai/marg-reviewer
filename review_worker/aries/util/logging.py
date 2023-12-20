import logging
import os
import pprint
from typing import Callable, List, Union

logger = logging.getLogger(__name__)


def init_logging(logfile=None, level=logging.INFO):
    handlers = [
        logging.StreamHandler(),
    ]
    if logfile:
        handlers.append(logging.FileHandler(logfile))

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] (%(name)s):  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        handlers=handlers,
    )


def pprint_metrics(metrics, print_fn: Union[Callable[[str], None], logging.Logger] = print, val_format="{:0.4f}", int_format="{:d}", name="eval"):
    if isinstance(print_fn, logging.Logger):
        print_fn = print_fn.info

    if name != "":
        name += " "

    for k, v in metrics.items():
        vstr = str(v)
        if isinstance(v, float) or isinstance(v, int):
            vstr = val_format.format(v)
            if isinstance(v, int) and int_format is not None:
                vstr = int_format.format(v)

        print_fn("{name}{metric_name}: {val}".format(name=name, metric_name=k, val=vstr))


class PrettyFloatPrinter(pprint.PrettyPrinter):
    def __init__(self, *args, **kwargs):
        if "sort_dicts" not in kwargs:
            kwargs["sort_dicts"] = False
        super().__init__(*args, **kwargs)

    def format(self, obj, ctx, maxlvl, lvl):
        if isinstance(obj, float):
            return "{:.4f}".format(obj), True, False
        # elif isinstance(obj, dict):
        #    print('gd', obj)
        #    v = '{' + ',\n'.join(["'{}': {}".format(k, self.format(v, ctx, maxlvl, lvl+1)[0]) for k, v in obj.items()]) + '}', True, False
        #    print(v[0])
        #    return v
        return pprint.PrettyPrinter.format(self, obj, ctx, maxlvl, lvl + 1)


def table2str(grid, format_fn=str, col_names=None, row_names=None, colsep=" | ", rowend="", header_row_sep="-"):
    if col_names is None:
        col_names = ["" for _ in range(len(grid[0]))]
    col_names = list(map(str, col_names))
    if row_names is None:
        row_names = ["" for _ in range(len(grid))]
    row_names = list(map(str, row_names))

    new_grid = [[""] + col_names]
    for rowidx, row in enumerate(grid):
        new_grid.append([row_names[rowidx]] + [format_fn(cell) for cell in row])
    return raw_table2str(new_grid, colsep=colsep, rowend=rowend, header_row_sep=header_row_sep)


def raw_table2str(grid, colsep=" | ", rowend="", header_row_sep="-"):
    s = ""

    col_widths = [max(len(grid[y][x]) for y in range(len(grid))) for x in range(len(grid[0]))]
    for y, row in enumerate(grid):
        if all(cell == "" for cell in row[1:]):
            continue
        # s += '    '
        s += colsep.join(["{text:>{width}s}".format(width=col_widths[x], text=cell) if col_widths[x] != 0 else "" for x, cell in enumerate(row)])
        s += "{}\n".format(rowend)
        if y == 0:
            if len(header_row_sep) == 1:
                s += header_row_sep * (sum(col_widths) + len(colsep) * (len(col_widths) - 1) + 1) + "\n"
            elif len(header_row_sep) == 0:
                continue
            else:
                s += header_row_sep + ("\n" if not header_row_sep.endswith("\n") else "")
    return s
