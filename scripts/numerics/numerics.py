#!/usr/bin/env python3
"""Numerical checks for aubio builds.

  record OUT.npz [--sounds DIR]
      run every algorithm on fixed signals and save every output

  compare A.npz B.npz [--ref REF.npz]
      how far B moved from A; with --ref (for instance a double-precision
      build of the same code), the error of A and of B against it

  accuracy [--json OUT.json]
      score onsets, beats, tempo and pitch against the generated signals'
      ground truth

Record runs at 44100 Hz (hop 512, window 2048) and at LedFx's 30000 Hz
(hop 500, window 4096). Inputs are float32 values whatever the build, so
a double-precision build records a reference for a float one.
"""

import argparse
import collections
import glob
import json
import os
import sys
import warnings

import numpy as np

import signals

CONFIGS = [(44100, 512, 2048), (30000, 500, 4096)]
ONSET_METHODS = ["energy", "hfc", "complex", "phase", "wphase", "specdiff",
                 "kl", "mkl", "specflux"]
PITCH_METHODS = ["yin", "yinfft", "yinfast", "mcomb", "fcomb", "schmitt",
                 "specacf"]
SPECDESC = ONSET_METHODS + ["centroid", "spread", "skewness", "kurtosis",
                            "slope", "decrease", "rolloff"]
WINDOWS = ["ones", "rectangle", "hamming", "hanning", "hanningz", "blackman",
           "blackman_harris", "gaussian", "welch", "parzen"]

# tolerances, as mir_eval's defaults
BEAT_TOLERANCE = 0.07
ONSET_TOLERANCE = 0.05
BEAT_TRIM = 5.0
TEMPO_TOLERANCE = 0.04
PITCH_TOLERANCE_CENTS = 50.0
LOCK_HOLD = 1.0


def load_aubio():
    import aubio
    return aubio


def as_input(aubio, x):
    # float32 values, in the build's sample type
    return np.asarray(x, dtype=np.float32).astype(aubio.float_type)


def frames(x, hop):
    for i in range(0, len(x) - hop + 1, hop):
        yield np.ascontiguousarray(x[i:i + hop])


# ---------------------------------------------------------------- record


class Recorder:
    def __init__(self, aubio):
        self.aubio = aubio
        self.out = {}

    def put(self, key, value):
        self.out[key] = np.asarray(value, dtype=np.float64)

    def signal(self, name, x, sr, hop, win):
        a = self.aubio
        tag = "%s@%d/%d/%d" % (name, sr, hop, win)
        for m in ONSET_METHODS:
            o = a.onset(m, win, hop, sr)
            desc, times = [], []
            for f in frames(x, hop):
                if o(f)[0]:
                    times.append(o.get_last())
                desc.append(o.get_descriptor())
            self.put(tag + "/onset/%s/desc" % m, desc)
            self.put(tag + "/onset/%s/times" % m, times)
        for m in PITCH_METHODS:
            p = a.pitch(m, win, hop, sr)
            p.set_unit("Hz")
            vals, conf = [], []
            for f in frames(x, hop):
                vals.append(p(f)[0])
                conf.append(p.get_confidence())
            self.put(tag + "/pitch/" + m, vals)
            self.put(tag + "/pitch/%s/conf" % m, conf)
        t = a.tempo("default", win, hop, sr)
        beats, bpm, conf = [], [], []
        for f in frames(x, hop):
            if t(f)[0]:
                beats.append(t.get_last())
            bpm.append(t.get_bpm())
            conf.append(t.get_confidence())
        self.put(tag + "/tempo/beats", beats)
        self.put(tag + "/tempo/bpm", bpm)
        self.put(tag + "/tempo/conf", conf)
        self.spectral(tag, x, sr, hop, win)
        n = a.notes("default", win, hop, sr)
        ev = [list(r) for r in (n(f) for f in frames(x, hop)) if r[0] or r[2]]
        self.put(tag + "/notes", ev if ev else np.zeros((0, 3)))
        for fname, fn in [("level_lin", a.level_lin), ("db_spl", a.db_spl),
                          ("zcr", a.zero_crossing_rate)]:
            self.put(tag + "/" + fname, [fn(f) for f in frames(x, hop)])
        self.put(tag + "/alpha_norm",
                 [a.alpha_norm(f, 2.) for f in frames(x, hop)])
        self.put(tag + "/silence",
                 [a.silence_detection(f, -70.) for f in frames(x, hop)])
        self.put(tag + "/level_detection",
                 [a.level_detection(f, -70.) for f in frames(x, hop)])
        for order, kind in [(7, "A"), (5, "C")]:
            flt = a.digital_filter(order)
            try:
                getattr(flt, "set_%s_weighting" % kind.lower())(sr)
            except ValueError:
                # before 0.4.12, only a list of samplerates had weightings
                continue
            self.put(tag + "/filter/" + kind,
                     np.concatenate([flt(f.copy()) for f in frames(x, hop)]))
        bq = a.digital_filter(3)
        bq.set_biquad(0.2, 0.3, 0.2, -0.5, 0.2)
        self.put(tag + "/filter/biquad",
                 np.concatenate([bq(f.copy()) for f in frames(x, hop)]))

    def spectral(self, tag, x, sr, hop, win):
        a = self.aubio
        pv, rpv, tpv = a.pvoc(win, hop), a.pvoc(win, hop), a.pvoc(win, hop)
        mfcc = a.mfcc(win, 40, 13, sr)
        sd = {k: a.specdesc(k, win) for k in SPECDESC}
        tss = a.tss(win, hop)
        fbs = {}
        for kind in ["mel", "htk", "slaney"]:
            fb = a.filterbank(40, win)
            if kind == "mel":
                fb.set_mel_coeffs(sr, 50, sr / 2.5)
            elif kind == "htk":
                fb.set_mel_coeffs_htk(sr, 50, sr / 2.5)
            else:
                fb.set_mel_coeffs_slaney(sr)
            self.put(tag + "/fbcoeffs/" + kind, fb.get_coeffs())
            fbs[kind] = fb
        fb = a.filterbank(40, win)
        fb.set_mel_coeffs(sr, 50, sr / 2.5)
        fb.set_power(2)
        fb.set_norm(0)
        fbs["power"] = fb
        acc = collections.defaultdict(list)
        for f in frames(x, hop):
            spec = pv(f)
            acc["norm"].append(spec.norm.copy())
            acc["phas"].append(spec.phas.copy())
            acc["resynth"].append(rpv.rdo(spec).copy())
            acc["mfcc"].append(mfcc(spec).copy())
            for k, d in sd.items():
                acc["sd/" + k].append(d(spec)[0])
            for k, fbk in fbs.items():
                acc["fb/" + k].append(fbk(spec).copy())
            tr, st = tss(tpv(f))
            acc["tss"].append(np.concatenate([tr.norm, st.norm]))
        for k, v in acc.items():
            self.put(tag + "/" + k, v)

    def functions(self):
        a = self.aubio
        for w in WINDOWS:
            for size in (512, 1000, 2048):
                self.put("window/%s/%d" % (w, size), a.window(w, size))
        f32 = lambda v: as_input(a, v)  # noqa: E731
        self.put("freqtomidi", a.freqtomidi(f32(np.linspace(0, 20000, 4001))))
        self.put("miditofreq", a.miditofreq(f32(np.linspace(-10, 140, 3001))))
        for name, fn, grid in [
                ("bintomidi", lambda b: a.bintomidi(b, 44100, 2048),
                 np.linspace(0, 1024, 513)),
                ("miditobin", lambda m: a.miditobin(m, 44100, 2048),
                 np.linspace(0, 127, 255)),
                ("bintofreq", lambda b: a.bintofreq(b, 44100, 2048),
                 np.linspace(0, 1024, 513)),
                ("freqtobin", lambda f: a.freqtobin(f, 44100, 2048),
                 np.linspace(0, 22050, 441)),
                ("hztomel", a.hztomel, np.linspace(0, 22050, 441)),
                ("meltohz", a.meltohz, np.linspace(0, 4000, 401)),
                ("hztomel_htk", a.hztomel_htk, np.linspace(0, 22050, 441)),
                ("meltohz_htk", a.meltohz_htk, np.linspace(0, 4000, 401))]:
            self.put(name, [fn(v) for v in f32(grid)])
        self.put("unwrap2pi", a.unwrap2pi(f32(np.linspace(-20, 20, 801))))
        rng = np.random.default_rng(3)
        for n in (8, 32, 100, 512):
            d = a.dct(n)
            v = f32(rng.standard_normal(n))
            self.put("dct/%d" % n, d(v))
            self.put("dct/%d/rdo" % n, d.rdo(v))


def read_sound(aubio, path, seconds):
    s = aubio.source(path, 0, 512)
    data = []
    while True:
        v, r = s()
        data.append(v[:r].copy())
        if r < 512:
            break
    return np.concatenate(data)[:int(s.samplerate * seconds)], s.samplerate


def record(args):
    aubio = load_aubio()
    rec = Recorder(aubio)
    for sr, hop, win in CONFIGS:
        for sig in signals.all_signals(sr):
            x = as_input(aubio, sig.x[:int(args.seconds * sr)])
            rec.signal(sig.name, x, sr, hop, win)
    if args.sounds:
        for path in sorted(glob.glob(os.path.join(args.sounds, "*.wav"))):
            x, sr = read_sound(aubio, path, args.seconds)
            if len(x) >= 4096:
                rec.signal(os.path.basename(path), x, sr, 512, 2048)
    rec.functions()
    np.savez_compressed(args.out, **rec.out)
    print("%d arrays from aubio %s (%s) in %s" % (
        len(rec.out), aubio.version, aubio.float_type, args.out))


# ---------------------------------------------------------------- compare


EVENTS = ("/times", "/beats", "/notes")


def is_event(key):
    return any(e in key for e in EVENTS)


def group(key):
    """the output a key belongs to, without the signal: 'onset/hfc/desc';
    the LedFx configuration gets its own groups"""
    parts = key.split("/")
    if "@" in parts[0]:
        prefix = "ledfx " if parts[0].endswith("@30000") else ""
        return prefix + "/".join(parts[3:])
    return parts[0]


def rel_error(x, r, key):
    """max and mean of |x - r|, relative to |r|; phases compared on the
    circle"""
    d = x - r
    if key.endswith("/phas"):
        d = (d + np.pi) % (2 * np.pi) - np.pi
    d = np.abs(d)
    d[np.isnan(x) & np.isnan(r)] = 0
    scale_max = max(float(np.nanmax(np.abs(r))), 1e-30)
    scale_mean = max(float(np.nanmean(np.abs(r))), 1e-30)
    return float(np.nanmax(d)) / scale_max, float(np.nanmean(d)) / scale_mean


def event_times(v, key):
    v = np.asarray(v)
    if key.endswith("/notes"):
        return v[:, 0] if v.size else v.ravel()
    return v.ravel()


def compare(args):
    a, b = np.load(args.a), np.load(args.b)
    ref = np.load(args.ref) if args.ref else None
    groups = collections.defaultdict(lambda: [0, 0., 0., 0., 0.])
    events = collections.defaultdict(lambda: [0, 0., 0.])
    missing = sorted(set(a) ^ set(b))
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        r = ref[k] if ref is not None and k in ref else x
        g = group(k)
        if is_event(k):
            # notes are midi numbers; times, samples: one hop is "the same"
            tol = 0.5 if k.endswith("/notes") else 512
            e = events[g]
            e[0] += 1
            e[1] += f_measure(event_times(x, k), event_times(r, k), tol)
            e[2] += f_measure(event_times(y, k), event_times(r, k), tol)
            continue
        if x.shape != y.shape or x.shape != r.shape or not r.size:
            if x.shape != y.shape:
                missing.append("%s: shape %s -> %s" % (k, x.shape, y.shape))
            continue
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if ref is not None:
                am, amean = rel_error(x, r, k)
            else:
                am, amean = 0., 0.
            bm, bmean = rel_error(y, r, k)
        s = groups[g]
        s[0] += 1
        s[1], s[2] = max(s[1], am), max(s[2], bm)
        s[3] += amean if np.isfinite(amean) else 0
        s[4] += bmean if np.isfinite(bmean) else 0
    if ref is not None:
        print("relative error against %s" % args.ref)
        print("%-30s %4s %10s %10s %10s %10s" % (
            "output", "n", "max A", "max B", "mean A", "mean B"))
    else:
        print("relative difference of %s from %s" % (args.b, args.a))
        print("%-30s %4s %10s %10s" % ("output", "n", "max", "mean"))
    for g, (n, am, bm, amean, bmean) in sorted(groups.items()):
        if ref is not None:
            amean, bmean = amean / n, bmean / n
            verdict = ""
            if bmean < 0.9 * amean:
                verdict = "better"
            elif bmean > 1.1 * amean + 1e-12:
                verdict = "worse"
            print("%-30s %4d %10.3g %10.3g %10.3g %10.3g  %s" % (
                g, n, am, bm, amean, bmean, verdict))
        else:
            print("%-30s %4d %10.3g %10.3g" % (g, n, bm, bmean / n))
    print()
    label = "A, B against the reference" if ref is not None else "B against A"
    print("%-30s %4s %8s %8s   F-measure of %s" % (
        "events", "n", "A", "B", label))
    for g, (n, fa, fb) in sorted(events.items()):
        print("%-30s %4d %8.4f %8.4f" % (g, n, fa / n, fb / n))
    for m in missing:
        print("only in one file, or reshaped:", m)


# ---------------------------------------------------------------- accuracy


def f_measure(estimated, reference, tolerance):
    """F-measure of one-to-one matches within tolerance (mir_eval's
    definition); matching sorted lists greedily is optimal in one
    dimension"""
    est, ref = np.sort(np.ravel(estimated)), np.sort(np.ravel(reference))
    if not est.size and not ref.size:
        return 1.0
    if not est.size or not ref.size:
        return 0.0
    i = j = hits = 0
    while i < est.size and j < ref.size:
        if abs(est[i] - ref[j]) <= tolerance:
            hits += 1
            i += 1
            j += 1
        elif est[i] < ref[j]:
            i += 1
        else:
            j += 1
    return 2.0 * hits / (est.size + ref.size)


def tempo_ok(bpm, truth, octave=False):
    ratios = (0.5, 1.0, 2.0) if octave else (1.0,)
    return np.any([np.abs(bpm - r * truth) <= TEMPO_TOLERANCE * r * truth
                   for r in ratios], axis=0)


def lock_times(t, bpm, sig):
    """for each tempo change, the time until the estimate is within
    tolerance (octave errors allowed) and stays there for LOCK_HOLD
    seconds; nan if it never does"""
    edges = getattr(sig, "edges", np.array([0.0, sig.duration]))
    ok = tempo_ok(bpm, sig.tempo(t), octave=True)
    out = []
    for start, end in zip(edges[:-1], edges[1:]):
        found = np.nan
        idx = np.flatnonzero((t >= start) & (t < end))
        for n, i in enumerate(idx):
            hold = idx[n:][t[idx[n:]] < t[i] + LOCK_HOLD]
            if t[i] + LOCK_HOLD <= end and ok[hold].all():
                found = t[i] - start
                break
        out.append(found)
    return out


def score_signal(aubio, sig, sr, hop, win):
    x = as_input(aubio, sig.x)
    res = {}
    if sig.onsets.size:
        for m in ONSET_METHODS:
            o = aubio.onset(m, win, hop, sr)
            times = [o.get_last_s() for f in frames(x, hop) if o(f)[0]]
            res["onset/" + m] = f_measure(times, sig.onsets, ONSET_TOLERANCE)
    if sig.beats.size:
        tr = aubio.tempo("default", win, hop, sr)
        beats, bpm = [], []
        for f in frames(x, hop):
            if tr(f)[0]:
                beats.append(tr.get_last_s())
            bpm.append(tr.get_bpm())
        beats, bpm = np.array(beats), np.array(bpm)
        res["beats"] = f_measure(beats[beats >= BEAT_TRIM],
                                 sig.beats[sig.beats >= BEAT_TRIM],
                                 BEAT_TOLERANCE)
        t = (np.arange(bpm.size) + 1) * hop / sr
        keep = t >= BEAT_TRIM
        truth = sig.tempo(t[keep])
        res["tempo"] = float(np.mean(tempo_ok(bpm[keep], truth)))
        res["tempo/octave"] = float(np.mean(
            tempo_ok(bpm[keep], truth, octave=True)))
        locks = lock_times(t, bpm, sig)
        res["lock/mean_s"] = float(np.nanmean(locks)) if not np.isnan(
            locks).all() else float("nan")
        res["lock/never"] = int(np.isnan(locks).sum())
    if sig.notes:
        for m in PITCH_METHODS:
            p = aubio.pitch(m, win, hop, sr)
            p.set_unit("Hz")
            est = np.array([p(f)[0] for f in frames(x, hop)])
            # the window ending at frame i's last sample is centred here
            t = ((np.arange(est.size) + 1) * hop - win / 2) / sr
            hits = total = 0
            for start, end, f0 in sig.notes:
                sel = est[(t >= start) & (t < end)]
                total += sel.size
                with np.errstate(divide="ignore", invalid="ignore"):
                    cents = 1200 * np.abs(np.log2(sel / f0))
                hits += int(np.sum(cents <= PITCH_TOLERANCE_CENTS))
            res["pitch/" + m] = hits / total if total else float("nan")
    return res


def print_scores(results):
    """one table per configuration: a row per metric, a column per signal"""
    configs = sorted({k.split("@")[1] for k in results}, reverse=True)
    for config in configs:
        names = sorted(k for k in results if k.endswith("@" + config))
        short = [n.split("@")[0] for n in names]
        width = max(8, max(len(s) for s in short))
        metrics = sorted({m for n in names for m in results[n]})
        print()
        print("%-16s " % config + " ".join("%*s" % (width, s) for s in short))
        for m in metrics:
            cells = []
            for n in names:
                v = results[n].get(m)
                cells.append("%*s" % (width, "" if v is None else (
                    "%d" % v if isinstance(v, int) else "%.3f" % v)))
            print("%-16s " % m + " ".join(cells))


def accuracy(args):
    aubio = load_aubio()
    results = {}
    for sr, hop, win in CONFIGS:
        for sig in signals.all_signals(sr):
            if not (sig.onsets.size or sig.beats.size or sig.notes):
                continue
            key = "%s@%d/%d/%d" % (sig.name, sr, hop, win)
            results[key] = score_signal(aubio, sig, sr, hop, win)
    print("aubio %s (%s)" % (aubio.version, aubio.float_type))
    print_scores(results)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"version": aubio.version, "float_type": aubio.float_type,
                       "results": results}, f, indent=1, sort_keys=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("record", help="save every output")
    p.add_argument("out")
    p.add_argument("--sounds", help="also run on the .wav files in this "
                   "directory (python/tests/sounds)")
    p.add_argument("--seconds", type=float, default=8.0,
                   help="length of each signal recorded (default 8)")
    p.set_defaults(run=record)
    p = sub.add_parser("compare", help="compare two recordings")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--ref", help="a reference recording, e.g. from a "
                   "double-precision build")
    p.set_defaults(run=compare)
    p = sub.add_parser("accuracy", help="score against ground truth")
    p.add_argument("--json", help="also write the scores here")
    p.set_defaults(run=accuracy)
    args = parser.parse_args(argv)
    warnings.simplefilter("ignore", UserWarning)
    args.run(args)


if __name__ == "__main__":
    sys.exit(main())
