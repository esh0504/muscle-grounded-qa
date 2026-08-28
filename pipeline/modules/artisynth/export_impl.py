# -*- coding: utf-8 -*-
# export_static_impl.py -- static PoseBank exporter (Step 3, ArtiSynth Jython).
#
# Maps each 11-D activation row of the Step-2 pool to a settled 3D tongue mesh:
# minimum-jerk ramp -> adaptive settle -> validity watchdog -> shard binaries.
#
# Design notes (kept from the development history):
#   1. SETTLING IS REQUIRED. A ramp that ends exactly at the last frame leaves the
#      tongue still moving -- not a static equilibrium.
#   2. TOPOLOGY IS WRITTEN ONCE. Per-sample OBJ export at scale (200k x 60 frames)
#      would be >1 TB with identical connectivity rewritten millions of times.
#   3. EVERY SAMPLE IS LABELLED. `except Exception` only catches crashes; element
#      inversion silently produces garbage. classify() labels each sample and
#      failures are KEPT (FailureBank) rather than dropped.
#
# Output ({out} = -Dstatic.out):
#   {out}/topology.obj / topology_info.txt
#   {out}/verts/shard_%05d.bin   float32 BIG-ENDIAN (n, n_surf_verts, 3)
#   {out}/nodes/shard_%05d.bin   float32 BIG-ENDIAN (n, n_fem_nodes, 3)
#   {out}/meta/shard_%05d.csv
# Loader: modules/mesh_io.py + modules/dorsum.py (numpy dtype '>f4')

from java.io import File, FileOutputStream, BufferedOutputStream, DataOutputStream
from java.io import BufferedReader, FileReader, FileWriter, BufferedWriter
from java.lang import System
from maspack.matrix import VectorNd
from maspack.interpolation.Interpolation import Order
from maspack.geometry.io import GenericMeshWriter
from artisynth.core.driver import Main
from artisynth.core.probes import NumericInputProbe

# ---------------------------------------------------------------- configuration
# No hardcoded machine paths: pool/out/model arrive as Java system properties
# (-Dstatic.pool / -Dstatic.out / -Dstatic.model), normally set by run_headless.bat.
# In the GUI console you can also call export_static_all(..., pool=..., out=...).
POOL_FILE = System.getProperty("static.pool")
OUT_DEFAULT = System.getProperty("static.out")
MODEL = System.getProperty("static.model",
                           "artisynth.models.tongue3d.StableFemMuscleTongueDemo")

# RAMP and SETTLE are NOT the same knob, and only one of them is safe to shorten.
#
#   RAMP_T = how fast the activation changes. Shortening this CHANGES THE PHYSICS:
#            bigger inertial transients, harder contact, more element inversion.
#            A gentle sample settles to the same pose anyway, but the high-effort
#            tail starts failing (minimum-jerk ramp requirement). DO NOT tune this
#            for speed. (ramp_sweep() below measures how far it can safely go, on
#            the WORST sample, not index 0.)
#
#   SETTLE = waiting for the transient to decay. Pure idle time.
#
# So the ramp is left alone and the settle is made ADAPTIVE: hold only until the
# tongue is actually still. Easy samples exit early, hard ones get as long as they
# need, and no accuracy is traded for speed.
# ramp_sweep() on the highest-effort sample (index 31189, activation sum 8.93):
#
#   ramp  peak_vel   min_elem_v   n_inv   max_dv_vs_ref
#   1.20     0.067       0.7230       0        --
#   0.60     0.139       0.7230       0     6.4e-14
#   0.40     0.212       0.7230       0     1.9e-13
#   0.30     0.280       0.7230       0     4.6e-13
#   0.20    76.8      -424352.1     143     0.779 m     <-- cliff
#   0.10    47.2       -30738.5     158     0.46  m
#
# Down to 0.30 the final pose is identical to machine precision. At 0.20 the ramp
# throws the tongue at the target instead of deforming it there: element volumes go
# NEGATIVE (inverted), 143 elements fold through themselves, and the pose lands
# 78 cm away. 0.40 keeps a 2x margin to that cliff -- 0.30 works but has none, and
# this is a single sample; total activation is not the only axis of difficulty.
# The compute difference between 0.6 and 0.4 is only ~11%, so the margin is cheap.
#
# Raised to 1.0 on purpose: a longer, gentler ramp deforms the tongue more slowly,
# which reduces inertial overshoot and therefore how often an element inverts. It
# costs a little more time per sample but yields more VALID samples and far less
# "Inverted elements" noise -- a good trade when the goal is a clean dataset.
RAMP_T = 1.0           # s, minimum-jerk ramp 0 -> a   (do NOT go below 0.3)
SETTLE_MIN = 0.2       # s, always hold at least this long after the ramp
SETTLE_MAX = 1.5       # s, give up waiting after this -> MARGINAL "not_settled"
CHECK_DT = 0.05        # s, polling interval == early-exit granularity
NRAMP_KNOTS = 24       # knots approximating the minimum-jerk profile

VEL_TOL = 1e-3         # m/s, max node speed below this => settled, stop early
VEL_MARGINAL = 5e-3    # m/s, still above this at SETTLE_MAX => MARGINAL
RAMP_PEAK_VEL = 0.5    # m/s, peak speed DURING the ramp above this => MARGINAL.
                       # A violent transient means the pose was reached by throwing
                       # the tongue at the target rather than deforming it there.
DIVERGE_VEL = 1e3      # m/s, abort -- the sample is already garbage

VOL_LO = 0.60          # global volume ratio bounds => INVALID_PHYSICAL outside
VOL_HI = 1.40
VOL_MARGIN_LO = 0.80
VOL_MARGIN_HI = 1.20
ELEM_VOL_MARGIN = 0.25 # min PER-ELEMENT volume ratio below this => MARGINAL

SHARD = 1000

DEFAULT_MUSCLES = ["GGP", "GGM", "GGA", "STY", "GH", "MH",
                   "HG", "VERT", "TRANS", "IL", "SL"]


# ------------------------------------------------------- inverted-element policy
# By default the FEM model both WARNS on every inverted-element step (spamming the
# console with identical stack traces) and, with adaptive stepping, keeps RETRYING
# the doomed step for the rest of the window. We handle inversion ourselves --
# simulate() breaks the instant getNumInverted() > 0 and classify() labels it
# INVALID_PHYSICAL -- so turn both the warning and the hard abort off. The mesh at
# first inversion is exactly what we want to keep in the FailureBank.
def configure_inversion(fem):
    try:
        fem.setWarnOnInvertedElements(False)
    except Exception, e:
        pass
    try:
        fem.setAbortOnInvertedElements(False)
    except Exception, e:
        pass


# ---------------------------------------------------------------- model access
def get_root():
    return Main.getMain().getRootModel()


def get_tongue():
    r = get_root()
    t = r.findComponent("models/mech/models/tongue")
    if t is None:
        t = r.findComponent("models/jawmodel/models/tongue")
    if t is None:
        t = r.models().get(0).models().get(0)
    return t


def clear_probes():
    r = get_root()
    names = []
    probes = r.getInputProbes()
    for i in range(probes.size()):
        nm = probes.get(i).getName()
        if nm is not None:
            names.append(nm)
    for nm in names:
        p = r.getInputProbes().get(nm)
        if p is not None:
            r.removeInputProbe(p)
    bps = []
    for w in r.getWayPoints():
        if w.isBreakPoint():
            bps.append(w)
    for w in bps:
        r.removeWayPoint(w)


# --------------------------------------------------------- minimum-jerk ramp
# s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5,  a(t) = s(t) * a*   (a0 = 0)
# Applied as dense LINEAR knots: ArtiSynth's Cubic probe interpolation is a
# different curve and overshoots on steep segments.
def minjerk(tau):
    return 10.0 * tau ** 3 - 15.0 * tau ** 4 + 6.0 * tau ** 5


def make_probes(muscle_names, vals):
    r = get_root()
    tongue = get_tongue()
    clear_probes()
    total = RAMP_T + SETTLE_MAX     # probe spans the worst case; simulate() exits early
    made = 0
    missing = []
    for i in range(len(muscle_names)):
        m = muscle_names[i]
        ex = tongue.getMuscleExciters().get(m)
        if ex is None:
            missing.append(m)
            continue
        ip = NumericInputProbe(ex, "excitation", 0, total)
        ip.setName(m)
        ip.setInterpolationOrder(Order.Linear)
        ip.setActive(True)
        for k in range(NRAMP_KNOTS + 1):
            tau = float(k) / NRAMP_KNOTS
            v = VectorNd(1)
            v.set(0, vals[i] * minjerk(tau))
            ip.addData(tau * RAMP_T, v)
        vh = VectorNd(1)            # HOLD
        vh.set(0, vals[i])
        ip.addData(total, vh)
        r.addInputProbe(ip)
        made = made + 1
    return made, missing, total


# ------------------------------------------------------------ validity watchdog
def node_state(fem):
    """(max node speed, has_nan)."""
    vmax = 0.0
    nodes = fem.getNodes()
    for i in range(nodes.size()):
        n = nodes.get(i)
        p = n.getPosition()
        if p.x != p.x or p.y != p.y or p.z != p.z:
            return vmax, 1
        s = n.getVelocity().norm()
        if s != s:
            return vmax, 1
        if s > vmax:
            vmax = s
    return vmax, 0


# Accessor names VERIFIED against this build via inspect_api():
#   FemModel3d  : getVolume(), getRestVolume(), getNumInverted()
#   FemElement3d: isInverted(), getVolume(), getRestVolume()
# (An earlier version guessed numInvertedElements()/getNumInvertedElements() --
#  both wrong, which is why n_inverted was NA on every row of the first probe run.)
def volume_ratio(fem):
    try:
        v0 = fem.getRestVolume()
        if v0 > 0:
            return fem.getVolume() / v0
    except Exception, e:
        pass
    return None


# THE REAL WATCHDOG SIGNAL.
# Global volume_ratio came out 0.997-1.014 on the first probe run -- reassuringly
# tight, and almost meaningless: near-incompressibility is enforced GLOBALLY, so
# total volume stays ~1.0 while an individual element folds through itself.
# Inversion is LOCAL. The first 20 samples already showed min per-element volume
# ratios down to 0.60 while the global ratio sat at 1.00.
def element_quality(fem):
    try:
        els = fem.getElements()
        min_ratio = 1e30
        n_inv = 0
        for i in range(els.size()):
            e = els.get(i)
            if e.isInverted():
                n_inv = n_inv + 1
            v0 = e.getRestVolume()
            if v0 > 0:
                r = e.getVolume() / v0
                if r < min_ratio:
                    min_ratio = r
        if min_ratio > 1e29:
            return None, n_inv
        return min_ratio, n_inv
    except Exception, e:
        return None, None


def classify(max_vel, nan, vol, ninv, emin, peak):
    if nan:
        return "FAILED_NUMERICAL", "nan_or_inf"
    if ninv is not None and ninv > 0:
        return "INVALID_PHYSICAL", "inverted_elements=%d" % ninv
    if emin is not None and emin <= 0.0:
        return "INVALID_PHYSICAL", "elem_vol_ratio=%.3f" % emin
    if vol is not None and (vol < VOL_LO or vol > VOL_HI):
        return "INVALID_PHYSICAL", "volume_ratio=%.3f" % vol
    if max_vel > VEL_MARGINAL:
        return "MARGINAL", "not_settled_vmax=%.4g" % max_vel
    reasons = []
    if peak > RAMP_PEAK_VEL:
        reasons.append("violent_ramp_peak=%.3g" % peak)
    if emin is not None and emin < ELEM_VOL_MARGIN:
        reasons.append("low_elem_vol=%.3f" % emin)
    if max_vel > VEL_TOL:
        reasons.append("slow_settle_vmax=%.4g" % max_vel)
    if vol is not None and (vol < VOL_MARGIN_LO or vol > VOL_MARGIN_HI):
        reasons.append("volume_drift=%.3f" % vol)
    if len(reasons) > 0:
        return "MARGINAL", ";".join(reasons)
    return "VALID", ""


# ---------------------------------------------------------------- simulate one
# ADAPTIVE SETTLE. Runs the minimum-jerk ramp untouched, then holds only as long as
# the tongue is still moving. Returns (max_vel, nan, peak_ramp_vel, settle_time).
#   - stops as soon as max node speed < VEL_TOL   -> easy samples cost much less
#   - gives up at SETTLE_MAX                      -> MARGINAL, not silently kept
#   - aborts on divergence                        -> no burning the window on garbage
#   - records peak speed DURING the ramp          -> catches "thrown into place" poses
def simulate(fem):
    main = Main.getMain()
    sched = main.getScheduler()
    sched.reset()
    total = RAMP_T + SETTLE_MAX
    t = 0.0
    peak = 0.0
    vmax = 0.0
    while t < total - 1e-9:
        t = min(t + CHECK_DT, total)
        sched.playRequest(t)
        main.waitForStop()
        # EARLY-OUT ON INVERSION. The moment any element inverts the sample is
        # already INVALID (hard failure). Without this the solver's adaptive
        # stepping keeps retrying the inverted step for the rest of the window --
        # spamming "Inverted elements" warnings and wasting seconds on a doomed
        # sample. Break immediately; classify() will label it INVALID.
        if fem.getNumInverted() > 0:
            return vmax, 0, peak, t - RAMP_T
        vmax, nan = node_state(fem)
        if nan:
            return vmax, 1, peak, t - RAMP_T
        if t <= RAMP_T + 1e-9:
            if vmax > peak:
                peak = vmax
            continue
        if vmax > DIVERGE_VEL:
            return vmax, 1, peak, t - RAMP_T
        st = t - RAMP_T
        if st >= SETTLE_MIN - 1e-9 and vmax < VEL_TOL:
            return vmax, 0, peak, st        # settled -> stop waiting
    return vmax, 0, peak, SETTLE_MAX


# --------------------------------------------------------------- binary writers
# append=False: a retried shard must OVERWRITE its .part, not append to it --
# otherwise a killed-and-restarted shard duplicates records and the binary no
# longer aligns with the meta csv.
def open_dos(path):
    return DataOutputStream(BufferedOutputStream(FileOutputStream(File(path), False)))


def mkdirs(path):
    d = File(path)
    if not d.exists():
        d.mkdirs()


def write_verts(dos, mesh):
    verts = mesh.getVertices()
    for i in range(verts.size()):
        p = verts.get(i).getPosition()
        dos.writeFloat(p.x)
        dos.writeFloat(p.y)
        dos.writeFloat(p.z)


def write_nodes(dos, fem):
    nodes = fem.getNodes()
    for i in range(nodes.size()):
        p = nodes.get(i).getPosition()
        dos.writeFloat(p.x)
        dos.writeFloat(p.y)
        dos.writeFloat(p.z)


def write_topology(out, fem, mesh):
    mkdirs(out)
    GenericMeshWriter.writeMesh(File("%s/topology.obj" % out), mesh)
    w = BufferedWriter(FileWriter(File("%s/topology_info.txt" % out)))
    w.write("n_surf_verts=%d\n" % mesh.getVertices().size())
    w.write("n_fem_nodes=%d\n" % fem.getNodes().size())
    w.write("n_elements=%d\n" % fem.getElements().size())
    w.write("dtype=>f4\n")
    w.write("ramp_t=%.4f settle_min=%.4f settle_max=%.4f\n" % (
        RAMP_T, SETTLE_MIN, SETTLE_MAX))
    w.close()
    print "topology: surf=%d nodes=%d elems=%d" % (
        mesh.getVertices().size(), fem.getNodes().size(), fem.getElements().size())


# ------------------------------------------------------------------ pool reader
def read_pool(filepath):
    br = BufferedReader(FileReader(filepath))
    muscle_names = None
    pool = {}
    line = br.readLine()
    while line is not None:
        s = line.strip()
        if s == "":
            line = br.readLine()
            continue
        if s.startswith("#"):
            if "keyframe" in s or "sequences" in s:
                break
            line = br.readLine()
            continue
        if s.startswith("index,"):
            muscle_names = s.split(",")[1:]
        else:
            parts = s.split(",")
            pool[int(parts[0])] = [float(x) for x in parts[1:]]
        line = br.readLine()
    br.close()
    if muscle_names is None:
        muscle_names = DEFAULT_MUSCLES
    return muscle_names, pool


# ------------------------------------------------------------------------ BATCH
# Shard-level resume: a shard counts as done only when its meta csv is closed, so a
# killed run redoes at most one shard.
#
# The pool puts REST + the single-muscle sweeps + the pairwise grids FIRST, then
# shuffles the rest. So any prefix contains all enumerated primitives AND is a
# balanced subsample -- you can stop at any point and still have a usable dataset,
# and the checkpoints give the coverage saturation curve for free.
#
# EMBARRASSINGLY PARALLEL: launch several ArtiSynth instances over disjoint ranges
#   export_static_all(0, 50000)   export_static_all(50000, 100000)   ...
# with the SAME out dir. They never touch the same shard, so they cannot collide.
def export_static_all(start=0, end=None, out=None, pool=None, resume=True):
    fp = pool if pool is not None else POOL_FILE
    od = out if out is not None else OUT_DEFAULT
    if fp is None:
        raise Exception("no pool file: pass pool=... or -Dstatic.pool=...")
    if od is None:
        raise Exception("no output dir: pass out=... or -Dstatic.out=...")
    muscle_names, pool_map = read_pool(fp)
    if end is None:
        end = max(pool_map.keys()) + 1
    tongue = get_tongue()
    configure_inversion(tongue)     # no warning spam, no retry-loop on inversion

    mkdirs(od)
    mkdirs("%s/verts" % od)
    mkdirs("%s/nodes" % od)
    mkdirs("%s/meta" % od)
    if not File("%s/topology.obj" % od).exists():
        write_topology(od, tongue, tongue.getSurfaceMesh())

    print "export_static_all: %d..%d  pool=%d  out=%s" % (start, end, len(pool_map), od)
    print "  ramp=%.2fs (minimum-jerk) + adaptive settle [%.2f..%.2f]s" % (
        RAMP_T, SETTLE_MIN, SETTLE_MAX)

    counts = {"VALID": 0, "MARGINAL": 0, "INVALID_PHYSICAL": 0, "FAILED_NUMERICAL": 0}
    sid = start // SHARD
    while sid * SHARD < end:
        lo = max(start, sid * SHARD)
        hi = min(end, (sid + 1) * SHARD)
        if resume and File("%s/meta/shard_%05d.csv" % (od, sid)).exists():
            print "  shard %05d  [skip]" % sid
            sid = sid + 1
            continue

        t0 = System.currentTimeMillis()
        dv = open_dos("%s/verts/shard_%05d.bin.part" % (od, sid))
        dn = open_dos("%s/nodes/shard_%05d.bin.part" % (od, sid))
        rows = []

        for index in range(lo, hi):
            if index not in pool_map:
                continue
            s0 = System.currentTimeMillis()
            label = "FAILED_NUMERICAL"
            reason = "exception"
            vmax = -1.0
            peak = -1.0
            st = -1.0
            vr = None
            ninv = None
            emin = None
            try:
                make_probes(muscle_names, pool_map[index])
                vmax, nan, peak, st = simulate(tongue)
                vr = volume_ratio(tongue)
                emin, ninv = element_quality(tongue)
                label, reason = classify(vmax, nan, vr, ninv, emin, peak)
            except Exception, e:
                reason = str(e).replace(",", ";")[:80]
            write_verts(dv, tongue.getSurfaceMesh())   # failures kept -> FailureBank
            write_nodes(dn, tongue)
            counts[label] = counts[label] + 1
            secs = (System.currentTimeMillis() - s0) / 1000.0
            svr = ("%.4f" % vr) if vr is not None else "NA"
            sni = ("%d" % ninv) if ninv is not None else "NA"
            sem = ("%.4f" % emin) if emin is not None else "NA"
            rows.append("%d,%s,%s,%.6g,%.6g,%s,%s,%s,%.3f,%.2f" % (
                index, label, reason, vmax, peak, svr, sni, sem, st, secs))

        dv.close()
        dn.close()
        w = BufferedWriter(FileWriter(File("%s/meta/shard_%05d.csv" % (od, sid))))
        w.write("index,label,reason,max_vel,peak_ramp_vel,vol_ratio,n_inverted,min_elem_vol,settle_t,secs\n")
        for r in rows:
            w.write(r + "\n")
        w.close()
        File("%s/verts/shard_%05d.bin.part" % (od, sid)).renameTo(
            File("%s/verts/shard_%05d.bin" % (od, sid)))
        File("%s/nodes/shard_%05d.bin.part" % (od, sid)).renameTo(
            File("%s/nodes/shard_%05d.bin" % (od, sid)))

        el = (System.currentTimeMillis() - t0) / 1000.0
        print "  shard %05d [%d..%d] %.1fs (%.2fs/sample)  V=%d M=%d I=%d F=%d" % (
            sid, lo, hi, el, el / max(1, hi - lo),
            counts["VALID"], counts["MARGINAL"],
            counts["INVALID_PHYSICAL"], counts["FAILED_NUMERICAL"])
        sid = sid + 1

    n = (counts["VALID"] + counts["MARGINAL"]
         + counts["INVALID_PHYSICAL"] + counts["FAILED_NUMERICAL"])
    print "DONE  n=%d" % n
    for k in ["VALID", "MARGINAL", "INVALID_PHYSICAL", "FAILED_NUMERICAL"]:
        print "  %-18s %6d  (%.1f%%)" % (k, counts[k], 100.0 * counts[k] / max(1, n))
    return counts


# ---------------------------------------------------------- API introspection
def inspect_api():
    fem = get_tongue()
    el = fem.getElements().get(0)
    print "--- FemModel3d ---"
    for m in sorted(dir(fem)):
        lm = m.lower()
        if "invert" in lm or "volume" in lm or "jacob" in lm or "detj" in lm:
            print "   ", m
    print "--- Element[0] ---"
    for m in sorted(dir(el)):
        lm = m.lower()
        if "invert" in lm or "volume" in lm or "jacob" in lm or "detj" in lm:
            print "   ", m
    print "integrator:", fem.getIntegrator()


# ------------------------------------------------- how fast can the ramp be?
# The settle no longer needs sweeping -- it is adaptive. The ramp is the one that
# is dangerous to shorten, because it changes the physics rather than just the
# waiting. So watch peak velocity, min element volume and inversion count, NOT just
# whether the final pose looks the same.
#
# Defaults to the HIGHEST-EFFORT sample in the pool. A gentle sample survives any
# ramp and tells you nothing; the failure mode only shows up on the aggressive ones.
def ramp_sweep(index=None, pool=None):
    global RAMP_T
    fp = pool if pool is not None else POOL_FILE
    if fp is None:
        raise Exception("no pool file: pass pool=... or -Dstatic.pool=...")
    muscle_names, pool_map = read_pool(fp)
    if index is None:
        best = -1.0
        for k in pool_map:
            s = sum(pool_map[k])
            if s > best:
                best = s
                index = k
        print "using highest-effort pool index %d (activation sum = %.2f)" % (index, best)
    vals = pool_map[index]
    tongue = get_tongue()
    configure_inversion(tongue)
    r0 = RAMP_T
    ref = None
    print "ramp_sweep on pool index %d" % index
    print "  %6s %7s %8s %10s %11s %6s %13s" % (
        "ramp", "secs", "settle", "peak_vel", "min_elem_v", "n_inv", "max_dv_vs_ref")
    for rt in [1.2, 0.6, 0.4, 0.3, 0.2, 0.1]:
        RAMP_T = rt
        t0 = System.currentTimeMillis()
        make_probes(muscle_names, vals)
        vmax, nan, peak, st = simulate(tongue)
        secs = (System.currentTimeMillis() - t0) / 1000.0
        emin, ninv = element_quality(tongue)
        verts = tongue.getSurfaceMesh().getVertices()
        cur = []
        for i in range(verts.size()):
            p = verts.get(i).getPosition()
            cur.append((p.x, p.y, p.z))
        dv = 0.0
        if ref is None:
            ref = cur
        else:
            for i in range(len(cur)):
                dx = cur[i][0] - ref[i][0]
                dy = cur[i][1] - ref[i][1]
                dz = cur[i][2] - ref[i][2]
                d = (dx * dx + dy * dy + dz * dz) ** 0.5
                if d > dv:
                    dv = d
        se = ("%.4f" % emin) if emin is not None else "NA"
        si = ("%d" % ninv) if ninv is not None else "NA"
        print "  %6.2f %7.2f %8.2f %10.3g %11s %6s %13.3g" % (
            rt, secs, st, peak, se, si, dv)
    RAMP_T = r0
    print ""
    print "Keep the LONGEST ramp that is still cheap. Stop shortening as soon as"
    print "peak_vel climbs, min_elem_v drops, n_inv goes non-zero, or max_dv exceeds"
    print "~1e-5 m. Past that point the ramp is throwing the tongue at the target"
    print "instead of deforming it there, and it will corrupt the high-effort tail"
    print "of the pool even though index 0 looks perfectly fine."


# ------------------------------------------- Stage 0: throughput + watchdog check
def probe_run(n=20, out=None, pool=None):
    if out is None and OUT_DEFAULT is None:
        raise Exception("no output dir: pass out=... or -Dstatic.out=...")
    od = out if out is not None else (OUT_DEFAULT + "_probe")
    export_static_all(0, n, out=od, pool=pool, resume=False)
    print ""
    print "Stage 0 checklist:"
    print "  1. seconds/sample x pool size = total compute budget"
    print "  2. meta csv: n_inverted / min_elem_vol must NOT be NA"
    print "  3. settle_t column shows how much the adaptive settle actually saved"


# ---------------------------------------------------------------------- startup
# Load the model only if one is not already present. In headless batch mode the
# model is preloaded on the command line (-model ...), so re-loading it here would
# be wasteful; in the GUI console nothing is loaded yet, so we load it.
def _model_loaded():
    try:
        r = Main.getMain().getRootModel()
        return r is not None and r.models().size() > 0
    except Exception, e:
        return False


if not _model_loaded():
    loadModel(MODEL)
clear_probes()
print "============================================"
print "  Static PoseBank Exporter"
print "============================================"
print "  model = %s" % MODEL
print "  pool  = %s" % str(POOL_FILE)
print "  out   = %s" % str(OUT_DEFAULT)
print "  ramp  = %.2fs minimum-jerk, adaptive settle [%.2f..%.2f]s" % (
    RAMP_T, SETTLE_MIN, SETTLE_MAX)
print ""
print "  ramp_sweep()                 # how fast can the ramp safely be? (worst sample)"
print "  probe_run(20)                # throughput + watchdog check"
print "  export_static_all(0, N)      # full run"
print ""
print "  !! Do NOT run the full batch in this GUI console: the timeline redraw"
print "  !! races with probe add/remove (ConcurrentModificationException) and can"
print "  !! kill the run. Use the headless runner instead:"
print "  !!   run_headless.bat 0 295157 4"
