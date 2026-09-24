# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Describe a compiled SDFG's ABI, so a language binding does not have to guess.

A compiled SDFG is called from outside DaCe -- by hand-written C, Fortran, Python or anything else
that can take a function pointer.  What such a caller needs to know is the argument list AND, for
each argument, what it is: an array or a scalar, of which dtype and rank, whether the kernel reads it
or writes it, and which runtime extents accompany it.

DaCe knows all of that (``SDFG.arglist()``, ``SDFG.arrays``, the access nodes) but exposes it only as
names -- ``compiled_sdfg.py`` reconstructs the rest internally.  So every external binding
reconstructed it by hand from the names, and every reconstruction is a place where a renamed argument
silently changes meaning.  Real examples of that heuristic, from a binding written against this
interface:

  * arrays vs scalars -- by a name PREFIX (``a.startswith(("ol", "or"))``);
  * an array's rank -- by special-casing a name (``if a == "cc"``);
  * an argument's intent -- by the same prefix;
  * an array's runtime extents -- by a regex on the name plus a hard-coded mapping of subscript to
    the caller's extent variable.

:func:`describe` reads the facts off the SDFG instead, where they are true by construction, and
:func:`write` persists them next to the compiled library as ``<name>.abi.json``.

This module deliberately knows nothing about any particular frontend or project: it works on
``SDFG`` objects and the filesystem.  ``dace_fortran.manifest`` re-exports it for the Fortran
bindings that were written against it first.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import dace
import dace.data as dt

# The C ABI names an array's runtime extents `<array>_d<k>`; this is how a flat argument list is split
# back into (arrays + scalars) and (extents).
EXTENT = re.compile(r'^(?P<base>.+)_d(?P<dim>\d+)$')


def _intents(sdfg) -> Dict[str, str]:
    """Read/written/both, per array, from the access nodes.

    An AccessNode's OUT edges mean its data is read by the consumer; its IN edges mean it is written.
    Both is a read-modify-write (an accumulator, ``rh = rh + ...``).  Non-transient arrays that never
    appear are reported as ``unknown`` rather than guessed at.
    """
    read, written = set(), set()
    for node, state in sdfg.all_nodes_recursive():
        if not isinstance(node, dace.nodes.AccessNode):
            continue
        if any(True for _ in state.out_edges(node)):
            read.add(node.data)
        if any(True for _ in state.in_edges(node)):
            written.add(node.data)
    out = {}
    for name, arr in sdfg.arrays.items():
        if arr.transient or isinstance(arr, dt.Scalar):
            continue
        if name in read and name in written:
            out[name] = "inout"
        elif name in read:
            out[name] = "in"
        elif name in written:
            out[name] = "out"
        else:
            out[name] = "unknown"
    return out


def describe(sdfg) -> Dict[str, Any]:
    """The library's ABI as data: argument order, and for each, its kind/dtype/rank/shape/intent."""
    # KEEP THE DESCRIPTORS, not just their names.  `arglist()` hands back a `dt.Data` per argument --
    # for a free symbol, `dt.Scalar(sdfg.symbols[name])`, carrying its dtype -- and taking only
    # `str(a)` threw that away, which is why the symbol branch below used to record "unknown" for a
    # fact it had been given.  See the note there.
    descriptors = sdfg.arglist()
    arglist = [str(a) for a in descriptors]

    # Split the flat list into arrays and scalars, and collect each array's extent arguments.  The
    # extents follow their array in the arglist, but the association is by NAME, not position, so a
    # reordering by DaCe cannot break it.
    arrays, scalars = [], []
    for a in arglist:
        desc = sdfg.arrays.get(a)
        if isinstance(desc, dt.Array) and not isinstance(desc, dt.Scalar):
            arrays.append(a)
        else:
            scalars.append(a)

    # `extents` is keyed by the ARRAY an extent belongs to, so membership in it says nothing about
    # whether an argument IS an extent -- that is `ext_names`.  Conflating the two puts every array
    # into the extent branch and crashes on `EXTENT.match(array)` being None.
    array_set = set(arrays)
    extents: Dict[str, List[str]] = {name: [] for name in arrays}
    ext_names = set()
    for a in arglist:
        m = EXTENT.match(a)
        if m and m.group("base") in array_set:
            extents[m.group("base")].append(a)
            ext_names.add(a)

    intents = _intents(sdfg)
    args = []
    for i, name in enumerate(arglist):
        desc = sdfg.arrays.get(name)
        if name in ext_names:
            m = EXTENT.match(name)
            args.append({
                "index": i,
                "name": name,
                "kind": "extent",
                "of": m.group("base"),
                "dim": int(m.group("dim")),
                "dtype": "int64",
            })
            continue
        if isinstance(desc, dt.Scalar):
            args.append({"index": i, "name": name, "kind": "scalar", "dtype": str(desc.dtype)})
            continue
        if isinstance(desc, dt.Array):
            args.append({
                "index": i,
                "name": name,
                "kind": "array",
                "dtype": str(desc.dtype),
                "rank": len(desc.shape),
                "shape": [str(s) for s in desc.shape],
                "intent": intents.get(name, "unknown"),
                "extents": extents[name],
                "is_view": isinstance(desc, dt.View),
            })
            continue
        # Not in `arrays` at all: a symbol, not a buffer.  It is still a real parameter of the
        # generated C function -- symbols reach the ABI with a concrete scalar type -- so record the
        # type `arglist()` gave us rather than "unknown".
        #
        # WHY THIS IS NOT PEDANTRY.  A binding that declares the C signature has to write a Fortran
        # type per argument, and "unknown" is not one: the consumer of this manifest, a Fortran port
        # of a CFD solver, could only work around it with a hardcoded list of the six loop-bound
        # names (`jb`, `je`, `kb`, `ke`, `lb`, `le`) that its kernels happen to use.  MEASURED in the
        # generated C for one of its libraries: those six are `int jb, int je, ...`, and the hardcoded
        # list asserted exactly that.  A seventh symbol, or a rename, would have fallen through to a
        # silent default instead.  The information was here the whole time.
        sdesc = descriptors.get(name)
        args.append({
            "index": i,
            "name": name,
            "kind": "symbol",
            "dtype": str(sdesc.dtype) if isinstance(sdesc, dt.Data) else "unknown",
        })

    return {
        "name": sdfg.name,
        "args": args,
        "order": arglist,
        "arrays": arrays,
        "scalars": scalars,
        "gpu": bool(sdfg.is_gpu_code()) if hasattr(sdfg, "is_gpu_code") else None,
    }


def write(sdfg, out_dir, name: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write ``<name>.abi.json`` next to a compiled library.

    Additive by design: DaCe's older argument-list artefacts are not touched, because they have
    readers.

    ``extra`` is the seam for the parts of an ABI that are the CALLER's knowledge and not the
    graph's -- in particular which expression at the call site each argument binds to.  It is
    deliberately a parameter rather than a built-in: `describe` reads the SDFG, so everything it
    reports is true of ANY compiled SDFG, while a binding says "this dummy is MFC's
    `dqL_prim_dx_n(1)%vf(iv%beg + 0)%sf`", which is a fact about one project.  A repository that
    knew that would be a general tool with one project's data model inside it.

    `extra` may not shadow a key `describe` produced: two descriptions of the same argument that
    disagree is exactly the kind of thing that is read as agreeing, and the graph's answer wins in
    the sense of being the one that cannot be stale.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or sdfg.name
    doc = describe(sdfg)
    if extra:
        clash = sorted(set(extra) & set(doc))
        if clash:
            raise ValueError(
                f"manifest: `extra` would overwrite {clash}, which `describe()` derives from the "
                f"SDFG.  A caller-supplied description that disagrees with the graph is worse than "
                f"no description -- put it under its own key.")
        doc = {**doc, **extra}
    path = out_dir / f"{name}.abi.json"
    path.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
    return path


def load(lib_dir, name: str) -> Dict[str, Any]:
    """Read a library's ABI manifest.  Raises with the fix, not a bare FileNotFoundError."""
    p = Path(lib_dir) / f"{name}.abi.json"
    if not p.exists():
        raise FileNotFoundError(
            f"no ABI manifest at {p} -- the library was compiled before manifest.describe() existed, "
            f"or by a different tree.  Rebuild it rather than guessing the ABI from the names.")
    return json.loads(p.read_text())
