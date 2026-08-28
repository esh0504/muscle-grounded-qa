# -*- coding: utf-8 -*-
# export_static_headless.py -- headless batch entry point (Step 3).
#
# Runs the static PoseBank export with NO GUI, so the timeline controller does not
# exist and cannot race with probe add/remove. (In the GUI console that race throws
# ConcurrentModificationException inside addInputProbe and can kill the run.)
#
# Everything is passed as Java system properties, so every parallel worker uses the
# SAME command line and differs only by -D flags (see run_headless.bat):
#
#   -Dstatic.impl=<path to export_static_impl.py>   (set by run_headless.bat)
#   -Dstatic.pool=<pool.txt from Step 2>
#   -Dstatic.out=<output dir>
#   -Dstatic.start=0  -Dstatic.end=295157

from java.lang import System as JSys

IMPL = JSys.getProperty("static.impl")
if IMPL is None:
    raise Exception("missing -Dstatic.impl=<path to export_static_impl.py> (use run_headless.bat)")
execfile(IMPL)

_start = int(JSys.getProperty("static.start", "0"))
_end = int(JSys.getProperty("static.end", "300000"))
_out = JSys.getProperty("static.out", OUT_DEFAULT)
_pool = JSys.getProperty("static.pool", POOL_FILE)
if _out is None or _pool is None:
    raise Exception("missing -Dstatic.out / -Dstatic.pool (use run_headless.bat)")

print ""
print ">>> headless batch: [%d, %d)  out=%s" % (_start, _end, _out)
export_static_all(_start, _end, out=_out, pool=_pool)

# In -noGui mode ArtiSynth stays alive after the script; exit so the worker frees
# its core when done.
Main.getMain().quit()
