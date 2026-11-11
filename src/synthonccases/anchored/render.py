"""Step two: turn retrieved anchors into a block of prompt text.

Separate from retrieval on purpose. What the model is TOLD about its anchors is at least as
open a question as which anchors it gets, and the two should be variable independently --
the same anchor set rendered three ways is a clean experiment, and the same rendering over
three anchor sets is a different one.

Renderers live in a registry so a new format is a new function, never an edit to a caller.
Each is a pure function of (corpus, retrieval result, policy) and returns text; none of them
touch the frozen prompt template, the schema, the enums or the conditioning spec. The block
produced here is ADDITIVE -- the frozen pipeline emits byte-identical prompts when no anchor
block is supplied.

Every renderer must state the support label and the anchor count in the text itself. A model
handed eleven anchors and one handed forty should be able to tell, and so should a human
reading the transcript afterwards.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .policy import AnchorPolicy, RetrievalResult
from .retrieval import Corpus
from .summary import MIN_STRATUM, choose_strata, format_km, km_summary, stratify

Renderer = Callable[[Corpus, RetrievalResult, AnchorPolicy], str]
_REGISTRY: dict[str, Renderer] = {}


def register(name: str) -> Callable[[Renderer], Renderer]:
    def deco(fn: Renderer) -> Renderer:
        if name in _REGISTRY:
            raise ValueError(f"renderer {name!r} is already registered")
        _REGISTRY[name] = fn
        return fn
    return deco


def render(name: str, corpus: Corpus, result: RetrievalResult,
           policy: AnchorPolicy) -> str:
    if name not in _REGISTRY:
        raise KeyError(f"unknown renderer {name!r}; have {sorted(_REGISTRY)}")
    if name == "none":
        # The unanchored arm is the frozen prompt for EVERY query, including the ones with
        # no anchors. Falling through to the refusal put "None available..." into 13 of the
        # R3 prompts, which is a sentence about retrieval in an arm that is supposed to know
        # nothing about retrieval -- and it made the control differ between queries.
        return ""
    if not result.has_anchors:
        return _refusal(result)
    return _REGISTRY[name](corpus, result, policy)


def available() -> list[str]:
    return sorted(_REGISTRY)


def _refusal(result: RetrievalResult) -> str:
    """What an unsupported query produces. Explicit, so silence is never mistaken for zero
    evidence being fine."""
    return ("## Comparable patients\n\n"
            f"None available. {result.note or 'Too few comparable patients were found.'}\n"
            "Answer from general knowledge and say that no comparable patients were "
            "available.\n")


def _fields(corpus: Corpus, policy: AnchorPolicy) -> list[str]:
    return list(policy.fields) if policy.fields else corpus.channels


def _anchor_rows(corpus: Corpus, result: RetrievalResult) -> pd.DataFrame:
    return corpus.frame.iloc[list(result.anchor_index)]


#: Plain-language gloss for every column that is not self-explanatory. A model handed
#: "dmets 1.00" has to guess; a clinician reading the transcript has to guess too.
#: How each extra time-to-event pair reads when the event happened, and when it did not.
_EXTRA_WORDS = {
    "overall survival": ("died", "alive at"),
    "next treatment": ("started another line", "no further line by"),
}

COLUMN_GLOSS = {
    "stage_iv": "stage IV at diagnosis (1 = yes)",
    "dmets": "distant metastases recorded at diagnosis (1 = yes)",
    "n_disease_sites": "distinct anatomical sites involved on the last scan before this "
                       "treatment started",
    "liver_met": "liver involvement seen on any scan before this treatment (1 = yes)",
    "brain_met": "brain involvement seen on any scan before this treatment (1 = yes)",
    "bone_met": "bone involvement seen on any scan before this treatment (1 = yes)",
    "last_scan_progressing": "the last scan before this treatment started was read as "
                             "progression (1 = yes)",
    "driver_bucket": "the driver gene on the sequencing panel resulted BEFORE this "
                     "treatment started ('none found' = panel done, no driver among the "
                     "genes it covers; 'not assessed' = no panel resulted by then)",
    "months_on_prior_line": "months the previous line of treatment lasted (0 = first line)",
    "prior_lines": "lines of systemic treatment before this one",
    "line_number": "which line of treatment this is",
    "treatment_start_year": "calendar year this treatment started",
    "age_at_start": "age in years when this treatment started",
    "outcome": "progression-free survival from the start of this treatment",
    "match": "how this patient qualified as comparable",
    "not known": "withheld: this patient's record stops at the time shown in `outcome`",
    "overall survival": "time from this treatment start until death",
    "next treatment": "time from this treatment start until the next line began",
}


def _header(corpus: Corpus, result: RetrievalResult, rows: pd.DataFrame,
            shown: int) -> str:
    """Neutral for every arm, with the facts computed from the rows rather than asserted.

    The same wording is used whether these are the nearest patients or a random control.
    An earlier version said "retrieved by similarity" and "same treatment" in both, which
    was true of one arm and false of the other -- a sentence the model could not verify and
    that we had no business writing. What IS true is stated instead: which cancer and which
    treatment these patients actually had, read off the rows. Where they are mixed, that is
    what appears, and the reader draws their own conclusion.
    """
    def _same(col: str) -> str | None:
        if col not in rows.columns:
            return None
        u = rows[col].astype(str).unique()
        return u[0] if len(u) == 1 else None

    cancer, action = _same("cancer"), _same(corpus.action_col)
    facts = []
    if cancer:
        facts.append(f"all have {cancer}")
    else:
        facts.append("they span several cancers")
    if action:
        facts.append(f"all received **{action}**")
    else:
        n = rows[corpus.action_col].nunique() if corpus.action_col in rows.columns else 0
        facts.append(f"they received {n} different treatments")

    spread = (f" Their spread around their own centre is {result.anchor_spread:.2f} in "
              f"standardised units."
              if result.anchor_spread == result.anchor_spread else "")
    cens = ("" if result.censored_fraction != result.censored_fraction else
            f" {result.censored_fraction:.0%} of them stopped being observed before "
            "progression, so their courses are shown only up to that point.")

    # HOW EACH ANCHOR QUALIFIED. A set of ten that is six strict and four relaxed is a
    # different piece of evidence from ten strict, and the model cannot weigh them
    # differently unless it is told which is which.
    tiers = ""
    if result.tier_counts and any(lab != "strict" for lab, _ in result.tier_counts):
        parts = []
        for lab, n in result.tier_counts:
            parts.append(f"{n} matched on everything" if lab == "strict"
                         else f"{n} matched more loosely ({lab})")
        listed = (parts[0] if len(parts) == 1
                  else " and ".join(parts) if len(parts) == 2
                  else ", ".join(parts[:-1]) + ", and " + parts[-1])
        tiers = (f"\n\nSupport was thin here: {listed}. The looser ones are "
                 "less comparable and should carry less weight.")
    elif result.tier_counts:
        tiers = (f"\n\nAll {result.n_strict} matched this patient on every clinical "
                 "criterion applied.")
    lead = (f"Summarising {result.n_anchors} real patients, with no individual patient "
            "listed, from the same care setting" if shown == 0 else
            f"Showing {shown} of {result.n_anchors} real patients from the same care "
            "setting")
    return (f"## Patient examples from the source corpus ({result.n_anchors})\n\n"
            f"{lead}: {', and '.join(facts)}. Every value below is as recorded at the "
            "moment that treatment started; nothing after it is shown except the outcome."
            f"{spread}\n\n"
            "They are EVIDENCE about what happens in this setting, not a rule to follow "
            "and not a guideline. Where they are a poor match for this patient, or where "
            "they disagree with each other, weigh them accordingly and use clinical "
            "judgement. Do not reproduce any single one, and do not treat their median as "
            "the answer."
            f"{cens}{tiers}\n\n")


def _choose_fields(corpus: Corpus, result: RetrievalResult,
                   policy: AnchorPolicy) -> tuple[list[str], list[str], list[str]]:
    """Which channels become columns, which collapse to a sentence, which are unobserved.

    Retrieval may look at everything; rendering must not. Three groups:

      columns      the mandatory context axes, then the channels that differ MOST between
                   this patient and the anchors -- the ones the model has to reason about.
      constant     channels identical across every anchor. Stated once in a sentence rather
                   than repeated down a column of the same value.
      unobserved   channels the source never records. Never mixed with the constants: "no
                   CNS disease" and "CNS was not assessed" are different claims.
    """
    if policy.fields:
        # Explicit fields are FIXED across arms. This matters more than it looks: ranking
        # columns by residual makes the column set depend on which anchors were retrieved,
        # so the shuffled control ended up differing from the treatment arm in its columns
        # as well as its anchors, and any difference between them became unattributable.
        rows = _anchor_rows(corpus, result)
        cols = [c for c in policy.fields if c in corpus.frame.columns]
        # constancy on the RAW values. Through `to_numeric` every text column became all
        # NaN, hence "one distinct value", hence constant -- so a categorical channel like
        # the driver gene could never reach the table however much it varied.
        constant = [c for c in cols
                    if rows[c].astype(str).nunique(dropna=False) <= 1]
        return [c for c in cols if c not in constant], constant, list(corpus.unobserved)
    rows = _anchor_rows(corpus, result)
    context = [c for c in policy.context_fields if c in corpus.frame.columns]
    pool = [c for c in corpus.channels
            if c not in context and c not in corpus.unobserved]
    constant, varying = [], []
    for c in pool:
        v = pd.to_numeric(rows[c], errors="coerce")
        (constant if v.nunique(dropna=False) <= 1 else varying).append(c)
    varying.sort(key=lambda c: -abs(result.residual.get(c, 0.0)))
    return context + varying[: policy.n_varying], constant, list(corpus.unobserved)


def _constant_sentence(rows: pd.DataFrame, constant: list[str],
                       unobserved: list[str]) -> str:
    out = []
    if constant:
        parts = [f"{c} = {_fmt(rows[c].iloc[0])}" for c in constant]
        out.append(f"Identical across all {len(rows)} anchors: " + "; ".join(parts) + ".")
    if unobserved:
        out.append("Not recorded by this source, for anchors or for this patient: "
                   + ", ".join(unobserved) + ".")
    return ("\n".join(out) + "\n") if out else ""


# ---------------------------------------------------------------------------- renderers ---
@register("compact")
def render_compact(corpus: Corpus, result: RetrievalResult, policy: AnchorPolicy) -> str:
    """The primary renderer: a table of only the columns that carry information.

    Mandatory context, then the channels this patient differs on most, then the outcome.
    Everything identical across the anchors is stated once; everything the source never
    records is stated separately, so a blank is never read as a zero.
    """
    rows = _anchor_rows(corpus, result)
    state_cols, constant, unobs = _choose_fields(corpus, result, policy)
    # display columns and state columns diverge here: "outcome" is a SYNTHESISED column that
    # exists only in the table, so the aggregate block must summarise the state columns and
    # never try to look "outcome" up in the frame
    cols = _outcome_columns(corpus, state_cols) if policy.include_outcomes else state_cols
    shown = rows.head(policy.render_target)
    # a per-row "match" column only when the anchors came in under different rules; adding
    # it to a set that is entirely strict would imply a distinction that does not exist
    mixed = len(set(result.tier_labels)) > 1
    head = (["match"] if mixed else []) + cols
    out = [_header(corpus, result, rows, len(shown)),
           "| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * len(head)) + "|"]
    for i, (_, r) in enumerate(shown.iterrows()):
        vals = [_cell(r, c, corpus) for c in cols]
        if mixed:
            vals = [result.tier_labels[i]] + vals
        out.append("| " + " | ".join(vals) + " |")
    out.append("")
    gloss = [f"- `{c}` — {COLUMN_GLOSS[c]}" for c in head if c in COLUMN_GLOSS]
    if gloss:
        out.append("Column meanings:\n" + "\n".join(gloss) + "\n")
    sentence = _constant_sentence(rows, constant, unobs)
    if sentence:
        out.append(sentence)
    if policy.include_aggregate:
        out.append(_aggregate_block(rows, corpus, policy, cols=state_cols))
    if policy.include_residual and result.residual:
        out.append(_residual_block(result, allowed=state_cols))
    if result.query_unobserved:
        out.append("\nNot assessed for this patient, and not to be inferred from its "
                   "absence: " + ", ".join(result.query_unobserved) + ".")
    return "\n".join(out)


@register("table")
def render_table(corpus: Corpus, result: RetrievalResult, policy: AnchorPolicy) -> str:
    """Every channel as a column. Kept as the un-curated comparator for the compact one."""
    rows = _anchor_rows(corpus, result)
    cols = _fields(corpus, policy)
    if policy.include_outcomes:
        cols = cols + [c for c in corpus.outcome_cols if c not in cols]
    shown = rows.head(policy.render_target)
    out = [_header(corpus, result, rows, len(shown)),
           "| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in shown.iterrows():
        out.append("| " + " | ".join(_cell(r, c, corpus) for c in cols) + " |")
    out.append("")
    if policy.include_aggregate:
        out.append(_aggregate_block(rows, corpus, policy))
    if policy.include_residual and result.residual:
        out.append(_residual_block(result, allowed=cols))
    return "\n".join(out)


@register("prose")
def render_prose(corpus: Corpus, result: RetrievalResult, policy: AnchorPolicy) -> str:
    """One short paragraph per anchor. Costs more tokens; reads like a case series."""
    rows = _anchor_rows(corpus, result)
    cols = _fields(corpus, policy)
    shown = rows.head(policy.render_target)
    out = [_header(corpus, result, rows, len(shown))]
    for i, (_, r) in enumerate(shown.iterrows(), start=1):
        facts = ", ".join(f"{c} {_cell(r, c, corpus)}" for c in cols)
        line = f"{i}. {facts}"
        if policy.include_outcomes and corpus.outcome_cols:
            line += ". Outcome: " + ", ".join(
                f"{c} {_fmt(r[c])}" for c in corpus.outcome_cols)
        out.append(line + ".")
    out.append("")
    if policy.include_aggregate:
        out.append(_aggregate_block(rows, corpus, policy))
    if policy.include_residual and result.residual:
        out.append(_residual_block(result, allowed=cols))
    return "\n".join(out)


@register("aggregate_only")
def render_aggregate_only(corpus: Corpus, result: RetrievalResult,
                          policy: AnchorPolicy) -> str:
    """No individual patients, only their summary.

    An ablation with a real question behind it: does the model need to see the patients, or
    only what they did on average? If this matches the table renderer, individual
    trajectories are costing tokens for nothing.
    """
    rows = _anchor_rows(corpus, result)
    out = [_header(corpus, result, rows, 0), _aggregate_block(rows, corpus, policy)]
    if policy.include_residual and result.residual:
        out.append(_residual_block(result, allowed=_fields(corpus, policy)))
    return "\n".join(out)


def _outcome_pair(corpus: Corpus, rows: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
    if not (corpus.outcome_time_col and corpus.outcome_event_col):
        return None
    return rows[corpus.outcome_time_col], rows[corpus.outcome_event_col]


def _query_value(rows: pd.DataFrame, result: RetrievalResult, axis: str) -> float | None:
    """The query's own value, recovered from the residual rather than passed in.

    residual = query - anchor centre, so query = centre + residual. Keeping the renderer a
    pure function of (corpus, result, policy) is worth this small arithmetic: a renderer
    that also took the query row could show the model something retrieval never saw.
    """
    if axis not in result.residual or axis not in rows.columns:
        return None
    centre = pd.to_numeric(rows[axis], errors="coerce").mean()
    return float(centre + result.residual[axis])


@register("aggregate_marginal")
def render_aggregate_marginal(corpus: Corpus, result: RetrievalResult,
                              policy: AnchorPolicy) -> str:
    """The matched cohort as ONE survival estimate. No individual patients.

    The form published evidence comes in, and the form this method has to work in when all
    that exists is a trial report. Read alone it is a subgroup rate: on this corpus 88% of
    the PFS spread survives matching on cancer, stage, line and regimen, so a marginal
    summary of a matched cohort is barely personalised. That is the point of running it as
    its own arm rather than assuming either way.
    """
    rows = _anchor_rows(corpus, result)
    out = [_header(corpus, result, rows, 0), ""]
    pair = _outcome_pair(corpus, rows)
    if policy.include_outcomes and pair is not None:
        out.append("**What happened to them**\n")
        out.append("- " + format_km(km_summary(*pair),
                                    label="All of these patients"))
        out.append("")
    out.append(_aggregate_block(rows, corpus, policy))
    if policy.include_residual and result.residual:
        out.append(_residual_block(result, allowed=_fields(corpus, policy)))
    if result.query_unobserved:
        out.append("\nNot assessed for this patient, and not to be inferred from its "
                   "absence: " + ", ".join(result.query_unobserved) + ".")
    return "\n".join(out)


@register("aggregate_conditional")
def render_aggregate_conditional(corpus: Corpus, result: RetrievalResult,
                                 policy: AnchorPolicy) -> str:
    """The matched cohort, then split on the axes this patient differs on most.

    A hard clinical match gives a coherent evidence population; it does not give a
    personalised one. What makes the forecast personal is where the query sits INSIDE that
    population, and this is the fixed-length way to say it: for each of a few pre-specified
    axes, what happened on each side of the split and which side this patient is on.

    One axis at a time, never crossed. Thirty patients cut two ways gives eight per cell,
    and a Kaplan-Meier median from eight patients is a number with no information in it.
    """
    rows = _anchor_rows(corpus, result)
    out = [_header(corpus, result, rows, 0), ""]
    pair = _outcome_pair(corpus, rows)
    if policy.include_outcomes and pair is not None:
        out.append("**What happened to them, overall**\n")
        out.append("- " + format_km(km_summary(*pair), label="All of these patients"))
        out.append("")
        # residuals put on a common scale FIRST. Raw, a difference of "8 years of age"
        # outranks "0.6 more disease sites" for no reason but the unit it is measured in,
        # and the strata would be chosen by whichever channel happens to have big numbers.
        scaled = {c: v / corpus.channel_sd(c) for c, v in result.residual.items()}
        axes = choose_strata(scaled, allowed=_fields(corpus, policy),
                             weights=corpus.prognostic_weights, k=policy.n_strata)
        lines: list[str] = []
        for axis in axes:
            qv = _query_value(rows, result, axis)
            if qv is None:
                continue
            groups = stratify(rows, {axis: qv}, axis)
            usable = [(lab, m, is_q) for lab, m, is_q in groups if m.sum() >= MIN_STRATUM]
            if len(usable) < 2:
                # a split that leaves one side nearly empty is not a comparison, and
                # printing it would suggest a contrast the data cannot support
                continue
            for lab, mask, is_query in usable:
                t, e = pair[0][mask], pair[1][mask]
                mark = "  <- this patient is in this group" if is_query else ""
                lines.append("- " + format_km(km_summary(t, e), label=lab) + mark)
            lines.append("")
        if lines:
            out.append("**Split on what this patient differs on most**\n")
            out.append("These splits were chosen from how far this patient sits from the "
                       "group centre on each axis, before seeing any outcome. They are "
                       "subgroups of the same small cohort, so read them as direction, not "
                       "as precise estimates.\n")
            out.extend(lines)
    out.append(_aggregate_block(rows, corpus, policy))
    if policy.include_residual and result.residual:
        out.append(_residual_block(result, allowed=_fields(corpus, policy)))
    if result.query_unobserved:
        out.append("\nNot assessed for this patient, and not to be inferred from its "
                   "absence: " + ", ".join(result.query_unobserved) + ".")
    return "\n".join(out)


@register("none")
def render_none(corpus: Corpus, result: RetrievalResult, policy: AnchorPolicy) -> str:
    """The unanchored arm. Retrieval still runs, so the two arms cover identical queries."""
    return ""


# ------------------------------------------------------------------------------ blocks ----
def _observed(rows: pd.DataFrame, corpus: Corpus, col: str) -> pd.Series:
    """The column with unassessed entries removed, so a summary never averages in a fill."""
    keep = pd.Series(True, index=rows.index)
    for flag, governed in corpus.observation_flags.items():
        if col in governed and flag in rows.columns:
            keep &= pd.to_numeric(rows[flag], errors="coerce").fillna(0) != 0
    return rows.loc[keep, col]


def _aggregate_block(rows: pd.DataFrame, corpus: Corpus, policy: AnchorPolicy,
                     cols: list[str] | None = None) -> str:
    cols = [c for c in (cols or _fields(corpus, policy))
            if c not in corpus.outcome_cols and c in rows.columns]
    out = [f"**All {len(rows)} comparable patients, summarised**", ""]
    for c in cols:
        seen = _observed(rows, corpus, c)
        v = pd.to_numeric(seen, errors="coerce").dropna()
        n_absent = len(rows) - len(v)
        if v.empty:
            # categorical. Blank entries are "not assessed", not a category called ""
            cat = seen.astype(str).replace({"": None, "nan": None}).dropna()
            if cat.empty:
                out.append(f"- {c}: not assessed in any of the {len(rows)}")
                continue
            top = cat.value_counts()
            missing = len(rows) - len(cat)
            tail = f", not assessed in {missing}" if missing else ""
            out.append(f"- {c}: {top.index[0]} in {top.iloc[0]} of {len(rows)}{tail}")
        else:
            tail = (f", {n_absent} not assessed" if n_absent else "")
            out.append(f"- {c}: median {v.median():.2f} "
                       f"(range {v.min():.2f}–{v.max():.2f}, from {len(v)} of "
                       f"{len(rows)}{tail})")
    if policy.include_outcomes:
        if corpus.outcome_time_col and corpus.outcome_event_col:
            t = pd.to_numeric(rows[corpus.outcome_time_col], errors="coerce")
            e = pd.to_numeric(rows[corpus.outcome_event_col], errors="coerce")
            prog = t[e == 1]
            # the median of a censored sample is not the median of the outcome, so the
            # censored share is stated beside it rather than folded into it
            if not prog.empty:
                out.append(f"- progressed in {int((e == 1).sum())} of {len(rows)}: median "
                           f"{prog.median():.1f} months, quartiles "
                           f"{prog.quantile(.25):.1f}–{prog.quantile(.75):.1f}")
            if int((e == 0).sum()):
                out.append(f"- for the other {int((e == 0).sum())} no progression was seen "
                           f"before observation stopped (median "
                           f"{t[e == 0].median():.1f} months); nothing later is shown for "
                           "them")
        else:
            for c in corpus.outcome_cols:
                v = pd.to_numeric(rows[c], errors="coerce").dropna()
                if not v.empty:
                    out.append(f"- {c}: median {v.median():.2f}, "
                               f"quartiles {v.quantile(.25):.2f}–{v.quantile(.75):.2f}")
    return "\n".join(out) + "\n"


def _residual_block(result: RetrievalResult, allowed: list[str] | None = None) -> str:
    """How the query differs from the anchor centre — the adaptation the model must make.

    Only channels that actually differ are listed. A residual of zero on fifteen channels is
    noise in the prompt, and burying the two that matter inside it is the opposite of help.
    """
    # the same exclusion the table applies. Dropping care-intensity channels from the
    # columns and then handing them back through the residual would put "n_prior_visits
    # +3.2" in front of the model after deciding it should not see visit counts at all.
    keep = set(allowed) if allowed else None
    # a difference on a channel nobody assessed for this patient is arithmetic on a fill
    # value: it would tell the model this patient has 1.4 fewer disease sites than the
    # group when no one counted their sites
    absent = set(result.query_unobserved)
    diffs = {k: v for k, v in result.residual.items()
             if abs(v) > 1e-9 and k not in absent and (keep is None or k in keep)}
    if not diffs:
        return ("**This patient versus those comparable patients**\n\n"
                "No material difference on the recorded channels.\n")
    ranked = sorted(diffs.items(), key=lambda kv: -abs(kv[1]))[:8]
    out = ["**This patient versus those comparable patients**", "",
           "Positive means this patient is higher than the comparable group's average.", ""]
    out += [f"- {k}: {v:+.2f}" for k, v in ranked]
    return "\n".join(out) + "\n"


def _outcome_columns(corpus: Corpus, cols: list[str]) -> list[str]:
    """Replace a (time, event) pair with one readable column; leave anything else alone."""
    if not (corpus.outcome_time_col and corpus.outcome_event_col):
        return cols + [c for c in corpus.outcome_cols if c not in cols]
    keep = [c for c in cols
            if c not in (corpus.outcome_time_col, corpus.outcome_event_col)]
    return keep + ["outcome"] + [lab for lab, _, _ in corpus.extra_outcomes]


def _outcome_text(row: pd.Series, corpus: Corpus) -> str:
    """"6.0 mo, progressed" or "no progression seen through 1.4 mo"."""
    t = pd.to_numeric(row.get(corpus.outcome_time_col), errors="coerce")
    e = pd.to_numeric(row.get(corpus.outcome_event_col), errors="coerce")
    if not np.isfinite(t):
        return "unknown"
    if e == 1:
        return f"{t:.1f} mo, progressed"
    # NOT "follow-up ended": this endpoint stopped being observable at t, which is usually
    # the last scan, and is not the same as the patient leaving the record. Saying follow-up
    # ended while another column shows them alive at 88 months is a contradiction the model
    # can see.
    return f"no progression seen through {t:.1f} mo"


def _censored_primary(row: pd.Series, corpus: Corpus) -> bool:
    if not (corpus.outcome_time_col and corpus.outcome_event_col):
        return False
    e = pd.to_numeric(row.get(corpus.outcome_event_col), errors="coerce")
    return bool(np.isfinite(e) and e == 0)


#: What an unassessed value reads as. A patient who never had a scan and a patient whose
#: scan found nothing are the same row of zeros in the matrix; on the page they must not be,
#: because the second is reassuring and the first is silence.
NOT_ASSESSED = "not assessed"


def _cell(row: pd.Series, col: str, corpus: Corpus) -> str:
    for flag, governed in corpus.observation_flags.items():
        if col in governed and flag in row.index and float(row.get(flag) or 0) == 0:
            return NOT_ASSESSED
    if col == "outcome" and corpus.outcome_time_col:
        return _outcome_text(row, corpus)
    for lab, tcol, ecol in corpus.extra_outcomes:
        if col == lab:
            # A patient whose primary endpoint is censored is one we stopped being able to
            # follow. Showing that they started another line, or when they died, hands back
            # exactly the future the censoring says we do not have -- and "no progression
            # through 0.7 mo" beside "started another line at 0.7 mo" tells the model they
            # progressed. Withheld rather than reconciled.
            if _censored_primary(row, corpus):
                return "not known"
            t = pd.to_numeric(row.get(tcol), errors="coerce")
            e = pd.to_numeric(row.get(ecol), errors="coerce")
            if not np.isfinite(t):
                return "unknown"
            done, open_ = _EXTRA_WORDS.get(lab, ("reached", "not reached by"))
            return f"{t:.1f} mo, {done}" if e == 1 else f"{t:.1f} mo, {open_} last contact"
    return _fmt(row[col])


def _fmt(v: object) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "—" if np.isnan(v) else f"{float(v):.2f}"
    return str(v)
