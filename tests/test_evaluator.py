"""Tests for the evaluator: evidence access, the agentic critic, and adjudication.

The headline test is `test_critic_can_reach_evidence_that_trail_hides` — it builds the
exact situation that motivated this work (a build error buried in the middle of a long
log) and shows the critic can now reach it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge.config import Budget, Config
from forge.evaluator import TrailTools, _apply_rulings, evaluate, solve
from forge.llm import Reply, Usage
from forge.loop import Result


# --------------------------------------------------------------------------- helpers
def tool_use(name, inp, id="t1"):
    return {"type": "tool_use", "name": name, "input": inp, "id": id}


def text(t):
    return {"type": "text", "text": t}


def cfg(tmp_path, **kw):
    c = Config(workspace=tmp_path / "ws", skills_dir=tmp_path / "sk",
               budget=Budget(max_turns=5, max_wall_seconds=30), max_eval_rounds=2)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


BURIED_ERROR = (
    "\n".join(f"compiling module_{i}.rs ... ok" for i in range(400))
    + "\nerror[E0433]: failed to resolve: use of undeclared crate `widget`\n"
    + "\n".join(f"compiling module_{i}.rs ... ok" for i in range(400, 800))
)


def result_with_buried_error(answer="the build succeeded cleanly"):
    return Result(
        answer=answer, turns=3, usage=Usage(), audit=[], stop="end_turn",
        tool_calls=[
            {"name": "run", "input": {"cmd": "cargo build"}, "output": BURIED_ERROR, "is_error": False},
            {"name": "read_file", "input": {"path": "Cargo.toml"}, "output": "[package]\nname='x'", "is_error": False},
        ],
    )


# --------------------------------------------------------------------------- evidence access
def test_trail_still_hides_the_error():
    """The motivating defect, pinned as a regression test."""
    r = result_with_buried_error()
    assert "error[E0433]" not in r.trail()      # the flat dump elides it
    assert len(r.trail()) < 2000                # ...from a 24k-char output


def test_grep_trail_finds_what_trail_hides():
    r = result_with_buried_error()
    hits = r.grep_trail(r"error\[E\d+\]")
    assert "E0433" in hits
    assert "[0] run" in hits                    # attributed to the right tool call


def test_grep_trail_searches_inputs_too():
    r = result_with_buried_error()
    assert "cargo build" in r.grep_trail("cargo")


def test_grep_trail_reports_absence_honestly():
    r = result_with_buried_error()
    assert "no matches" in r.grep_trail("definitely-not-present-xyz")


def test_grep_trail_rejects_bad_regex():
    r = result_with_buried_error()
    with pytest.raises(ValueError, match="bad regex"):
        r.grep_trail("[unclosed")


def test_tool_output_returns_full_text_paginated():
    r = result_with_buried_error()
    first = r.tool_output(0, offset=0, limit=5000)
    assert "chars 0-5000 of 24," in first
    assert "more chars" in first                # tells the critic how to continue
    # paging through reaches the buried error, which truncation never would
    found = any("E0433" in r.tool_output(0, offset=o, limit=5000) for o in range(0, 25000, 5000))
    assert found


def test_tool_output_bounds_checked():
    r = result_with_buried_error()
    with pytest.raises(IndexError, match="no tool call at index 9"):
        r.tool_output(9)


def test_trail_index_is_bounded_and_complete():
    r = result_with_buried_error()
    idx = r.trail_index()
    assert "[0] run" in idx and "[1] read_file" in idx   # every call listed
    assert "24,755 chars" in idx                          # size disclosed
    assert len(idx) < 1000                                # but stays small


def test_trail_index_flags_errors():
    r = Result("a", 1, Usage(), [], "end_turn",
               tool_calls=[{"name": "run", "input": {}, "output": "boom", "is_error": True}])
    assert "!ERROR" in r.trail_index()


def test_trailtools_dispatch_surfaces_errors_not_crashes():
    tt = TrailTools(result_with_buried_error())
    out, err = tt.dispatch("read_tool_output", {"index": 99})
    assert err is True and "no tool call" in out
    out, err = tt.dispatch("nonsense", {})
    assert err is True and "unknown tool" in out
    assert tt.calls == 2


# --------------------------------------------------------------------------- the critic
def test_critic_can_reach_evidence_that_trail_hides(tmp_path):
    """The whole point: a critic that investigates catches the lie a truncated dump hides."""
    r = result_with_buried_error(answer="the build succeeded cleanly, all modules compiled")
    seen: dict = {}

    class Critic:
        def __init__(self):
            self.n = 0

        def call(self, **kw):
            self.n += 1
            if self.n == 1:
                seen["prompt"] = kw["messages"][0]["content"]
                return Reply([tool_use("grep_trail", {"pattern": r"error|warning"}, "g1")],
                             "tool_use", Usage(1, 1))
            if self.n == 2:
                # what came back from grep is in the tool_result
                seen["grep_result"] = kw["messages"][-1]["content"][0]["content"]
                return Reply([tool_use("submit_verdict", {
                    "verdict": "revise", "score": 2,
                    "critique": "build actually failed with E0433",
                    "required_fixes": ["fix the undeclared crate `widget` and rebuild"],
                    "unsupported_claims": ["'the build succeeded cleanly'"],
                }, "s1")], "tool_use", Usage(1, 1))
            # adjudicator upholds
            return Reply([tool_use("submit_rulings", {"rulings": [
                {"claim": "'the build succeeded cleanly'", "ruling": "unsupported",
                 "evidence": "tool 0 shows error[E0433]"}]}, "r1")], "tool_use", Usage(1, 1))

    v = evaluate(cfg(tmp_path), Critic(), "build the project", r)
    # the critic was given an index, not the contents
    assert "error[E0433]" not in seen["prompt"]
    # but grep reached the truth
    assert "E0433" in seen["grep_result"]
    assert v.verdict == "revise" and v.score == 2
    assert v.unsupported_claims == ["'the build succeeded cleanly'"]
    assert v.checks == 1


def test_critic_prose_gets_nudged_to_submit(tmp_path):
    class Waffler:
        def __init__(self):
            self.n = 0

        def call(self, **kw):
            self.n += 1
            if self.n == 1:
                return Reply([text("Let me think about this...")], "end_turn", Usage(1, 1))
            return Reply([tool_use("submit_verdict", {
                "verdict": "pass", "score": 9, "critique": "fine",
                "required_fixes": [], "unsupported_claims": []})], "tool_use", Usage(1, 1))

    v = evaluate(cfg(tmp_path), Waffler(), "task", result_with_buried_error())
    assert v.passed


def test_critic_that_never_submits_is_forced(tmp_path):
    """Runs out of investigation turns -> we force a verdict rather than guessing."""
    class Stubborn:
        def __init__(self):
            self.forced = False

        def call(self, **kw):
            if kw.get("tool_choice", {}).get("name") == "submit_verdict":
                self.forced = True
                return Reply([tool_use("submit_verdict", {
                    "verdict": "pass", "score": 8, "critique": "forced",
                    "required_fixes": [], "unsupported_claims": []})], "tool_use", Usage(1, 1))
            return Reply([tool_use("grep_trail", {"pattern": "x"}, "g")], "tool_use", Usage(1, 1))

    llm = Stubborn()
    c = cfg(tmp_path, critic_max_turns=3)
    v = evaluate(c, llm, "task", result_with_buried_error())
    assert llm.forced and v.verdict == "pass"


# --------------------------------------------------------------------------- adjudication
def _critic_then_adjudicator(verdict_payload, rulings_payload):
    """LLM that plays critic first, then adjudicator, routing on the tools offered."""
    class Both:
        def call(self, **kw):
            names = {t.get("name") for t in (kw.get("tools") or [])}
            if "submit_rulings" in names:
                return Reply([tool_use("submit_rulings", rulings_payload, "r")], "tool_use", Usage(1, 1))
            return Reply([tool_use("submit_verdict", verdict_payload, "s")], "tool_use", Usage(1, 1))
    return Both()


def test_adjudicator_drops_false_accusation(tmp_path):
    """A critic that cries wolf gets overruled before the agent ever hears it."""
    llm = _critic_then_adjudicator(
        {"verdict": "revise", "score": 5, "critique": "you never ran the build",
         "required_fixes": ["run cargo build"],
         "unsupported_claims": ["'I ran cargo build'"]},
        {"rulings": [{"claim": "'I ran cargo build'", "ruling": "supported",
                      "evidence": "tool 0 is literally `cargo build`"}]},
    )
    v = evaluate(cfg(tmp_path), llm, "build it", result_with_buried_error())
    assert v.unsupported_claims == []                     # dropped
    assert v.dropped_claims == ["'I ran cargo build'"]    # but recorded
    assert v.verdict == "revise"                          # other fix still stands


def test_adjudicator_upholds_true_accusation(tmp_path):
    llm = _critic_then_adjudicator(
        {"verdict": "revise", "score": 2, "critique": "build failed",
         "required_fixes": ["fix E0433"], "unsupported_claims": ["'the build succeeded'"]},
        {"rulings": [{"claim": "'the build succeeded'", "ruling": "unsupported",
                      "evidence": "tool 0 shows error[E0433]"}]},
    )
    v = evaluate(cfg(tmp_path), llm, "build it", result_with_buried_error())
    assert v.unsupported_claims == ["'the build succeeded'"]
    assert v.dropped_claims == []


def test_unclear_ruling_keeps_the_claim(tmp_path):
    """Benefit of the doubt goes to raising the concern, not burying it."""
    llm = _critic_then_adjudicator(
        {"verdict": "revise", "score": 4, "critique": "c", "required_fixes": ["f"],
         "unsupported_claims": ["'tests pass'"]},
        {"rulings": [{"claim": "'tests pass'", "ruling": "unclear", "evidence": "ambiguous"}]},
    )
    v = evaluate(cfg(tmp_path), llm, "t", result_with_buried_error())
    assert v.unsupported_claims == ["'tests pass'"]


def test_missing_ruling_keeps_the_claim(tmp_path):
    """Adjudicator silence must not silently exonerate."""
    llm = _critic_then_adjudicator(
        {"verdict": "revise", "score": 4, "critique": "c", "required_fixes": ["f"],
         "unsupported_claims": ["'a'", "'b'"]},
        {"rulings": [{"claim": "'a'", "ruling": "supported", "evidence": "e"}]},  # 'b' unruled
    )
    v = evaluate(cfg(tmp_path), llm, "t", result_with_buried_error())
    assert v.unsupported_claims == ["'b'"]
    assert v.dropped_claims == ["'a'"]


def test_adjudication_can_be_disabled(tmp_path):
    llm = _critic_then_adjudicator(
        {"verdict": "revise", "score": 4, "critique": "c", "required_fixes": ["f"],
         "unsupported_claims": ["'x'"]},
        {"rulings": [{"claim": "'x'", "ruling": "supported", "evidence": "e"}]},
    )
    v = evaluate(cfg(tmp_path, adjudicate=False), llm, "t", result_with_buried_error())
    assert v.unsupported_claims == ["'x'"]   # not adjudicated, so not dropped
    assert v.dropped_claims == []


def test_text_fallback_is_exact_only():
    """Kept for models that ignore the schema; exact-match modulo case/quotes/space."""
    claims = ["'the build succeeded cleanly'"]
    r, _ = _apply_rulings({"rulings": [
        {"claim": "The Build Succeeded Cleanly", "ruling": "supported", "evidence": "e"}]}, claims)
    assert r == {claims[0]: "supported"}


# --------------------------------------------------------------------------- coherence guards
def test_revise_with_no_substance_becomes_pass(tmp_path):
    """A reviewer who sends work back must say why. This one didn't."""
    llm = _critic_then_adjudicator(
        {"verdict": "revise", "score": 6, "critique": "vibes are off",
         "required_fixes": [], "unsupported_claims": []},
        {"rulings": []},
    )
    v = evaluate(cfg(tmp_path), llm, "t", result_with_buried_error())
    assert v.passed
    assert "no actionable fixes" in v.override


def test_all_claims_dropped_and_no_fixes_becomes_pass(tmp_path):
    """Critic's only objection evaporated under adjudication -> nothing to send back for."""
    llm = _critic_then_adjudicator(
        {"verdict": "revise", "score": 5, "critique": "you never ran it",
         "required_fixes": [], "unsupported_claims": ["'I ran it'"]},
        {"rulings": [{"claim": "'I ran it'", "ruling": "supported", "evidence": "tool 0"}]},
    )
    c = cfg(tmp_path)
    v = evaluate(c, llm, "t", result_with_buried_error())
    assert v.passed
    assert "failed adjudication" in v.override
    assert v.score >= c.pass_score


def test_real_fixes_survive_the_guard(tmp_path):
    """The guard must not launder a legitimate revise into a pass."""
    llm = _critic_then_adjudicator(
        {"verdict": "revise", "score": 3, "critique": "broken",
         "required_fixes": ["fix E0433"], "unsupported_claims": []},
        {"rulings": []},
    )
    v = evaluate(cfg(tmp_path), llm, "t", result_with_buried_error())
    assert v.verdict == "revise" and v.override is None


def test_malformed_verdict_does_not_crash(tmp_path):
    class Junk:
        def call(self, **kw):
            return Reply([tool_use("submit_verdict", {}, "s")], "tool_use", Usage(1, 1))

    v = evaluate(cfg(tmp_path), Junk(), "t", result_with_buried_error())
    assert v.verdict == "pass" and v.override    # empty revise -> guard fires
    assert v.score >= 1


# --------------------------------------------------------------------------- integration
def test_only_surviving_claims_reach_the_agent(tmp_path):
    """End to end: the false accusation never appears in the agent's retry prompt."""
    prompts: list[str] = []

    class Full:
        def __init__(self):
            self.round = 0

        def call(self, **kw):
            names = {t.get("name") for t in (kw.get("tools") or [])}
            if "submit_rulings" in names:
                return Reply([tool_use("submit_rulings", {"rulings": [
                    {"claim": "'I cloned the repo'", "ruling": "supported", "evidence": "tool 0"},
                    {"claim": "'the build succeeded'", "ruling": "unsupported", "evidence": "E0433"},
                ]}, "r")], "tool_use", Usage(1, 1))
            if "submit_verdict" in names:
                self.round += 1
                if self.round == 1:
                    return Reply([tool_use("submit_verdict", {
                        "verdict": "revise", "score": 3, "critique": "build broke",
                        "required_fixes": ["fix E0433"],
                        "unsupported_claims": ["'I cloned the repo'", "'the build succeeded'"],
                    }, "s")], "tool_use", Usage(1, 1))
                return Reply([tool_use("submit_verdict", {
                    "verdict": "pass", "score": 9, "critique": "fixed",
                    "required_fixes": [], "unsupported_claims": []}, "s")], "tool_use", Usage(1, 1))
            prompts.append(str(kw["messages"][0]["content"]))
            return Reply([text("attempt")], "end_turn", Usage(1, 1))

    c = cfg(tmp_path, max_eval_rounds=3)
    out = solve(c, Full(), "build the project")
    assert out.verdict.passed and out.rounds == 2
    retry = prompts[1]
    assert "'the build succeeded'" in retry        # the real accusation was passed on
    assert "'I cloned the repo'" not in retry      # the false one was suppressed


# --------------------------------------------------------------------------- claim mapping
# The original defect: rulings were matched back to claims by substring containment.
# A non-verbatim echo of a claim whose text was a prefix of another landed on the WRONG
# claim, transposing rulings silently. Which claim got dropped depended on dict insertion
# order. These tests pin the fix and the old behaviour's impossibility.

PREFIX_CLAIMS = ["the build succeeded", "The build succeeded cleanly on all 800 modules"]


def test_containment_matching_cannot_recur():
    """Claim 1's text is a prefix of claim 2's. A case-changed echo of claim 2 used to
    match claim 1 by containment; exact-normalised matching gets it right."""
    r, notes = _apply_rulings({"rulings": [
        {"claim": "the build succeeded cleanly on all 800 modules",  # claim 2, 'The'->'the'
         "ruling": "supported", "evidence": "e"},
    ]}, PREFIX_CLAIMS)
    assert r == {PREFIX_CLAIMS[1]: "supported"}   # NOT claim 0
    assert PREFIX_CLAIMS[0] not in r


def test_id_mapping_is_unambiguous():
    r, notes = _apply_rulings({"rulings": [
        {"claim_id": 2, "ruling": "unsupported", "evidence": "E0433 at tool 0"},
        {"claim_id": 1, "ruling": "supported", "evidence": "cargo build did run"},
    ]}, PREFIX_CLAIMS)
    assert r == {PREFIX_CLAIMS[1]: "unsupported", PREFIX_CLAIMS[0]: "supported"}
    assert notes == []


def test_id_takes_precedence_over_text():
    """If a model sends both and they disagree, the id is the contract."""
    r, _ = _apply_rulings({"rulings": [
        {"claim_id": 2, "claim": "the build succeeded", "ruling": "supported", "evidence": "e"},
    ]}, PREFIX_CLAIMS)
    assert r == {PREFIX_CLAIMS[1]: "supported"}


def test_reworded_claim_yields_a_note_not_a_guess():
    r, notes = _apply_rulings({"rulings": [
        {"claim": "the build was reported as successful", "ruling": "supported", "evidence": "e"},
    ]}, PREFIX_CLAIMS)
    assert r == {}                                        # nothing dropped on a guess
    assert any("matched nothing" in n for n in notes)
    assert any("stand unchallenged" in n for n in notes)  # both claims survive


# --- every rejection path must leave the claim standing -----------------------
@pytest.mark.parametrize("bad,expect", [
    ({"claim_id": 99, "ruling": "supported", "evidence": "e"}, "out of range"),
    ({"claim_id": "two", "ruling": "supported", "evidence": "e"}, "not an integer"),
    ({"claim_id": 1, "ruling": "definitely-fine", "evidence": "e"}, "unknown verdict"),
    ({"claim_id": 0, "ruling": "supported", "evidence": "e"}, "out of range"),
])
def test_malformed_rulings_are_discarded_and_noted(bad, expect):
    r, notes = _apply_rulings({"rulings": [bad]}, PREFIX_CLAIMS)
    assert r == {}                                  # claim survives — the safe direction
    assert any(expect in n for n in notes)


def test_non_object_ruling_discarded():
    r, notes = _apply_rulings({"rulings": ["just a string", 42]}, PREFIX_CLAIMS)
    assert r == {}
    assert sum("not an object" in n for n in notes) == 2


def test_duplicate_consistent_rulings_are_fine():
    r, notes = _apply_rulings({"rulings": [
        {"claim_id": 1, "ruling": "supported", "evidence": "a"},
        {"claim_id": 1, "ruling": "supported", "evidence": "b"},
    ]}, PREFIX_CLAIMS)
    assert r[PREFIX_CLAIMS[0]] == "supported"
    assert not any("conflicting" in n for n in notes)


def test_conflicting_rulings_distrust_both():
    """An adjudicator that says two things about one claim gets believed on neither."""
    r, notes = _apply_rulings({"rulings": [
        {"claim_id": 1, "ruling": "supported", "evidence": "a"},
        {"claim_id": 1, "ruling": "unsupported", "evidence": "b"},
    ]}, PREFIX_CLAIMS)
    assert PREFIX_CLAIMS[0] not in r                # claim stands
    assert any("conflicting" in n and "claim 1" in n for n in notes)


def test_missing_rulings_are_reported():
    r, notes = _apply_rulings({"rulings": [
        {"claim_id": 1, "ruling": "supported", "evidence": "a"},
    ]}, PREFIX_CLAIMS)
    assert any("claim(s) [2]" in n for n in notes)


def test_empty_rulings_leaves_everything_standing():
    r, notes = _apply_rulings({"rulings": []}, PREFIX_CLAIMS)
    assert r == {}
    assert any("[1, 2]" in n for n in notes)


# --- integration --------------------------------------------------------------
def test_prefix_claims_ruled_correctly_end_to_end(tmp_path):
    """The full path: critic flags two overlapping claims, adjudicator rules by id,
    only the true accusation reaches the agent."""
    llm = _critic_then_adjudicator(
        {"verdict": "revise", "score": 3, "critique": "build broke",
         "required_fixes": ["fix E0433"],
         "unsupported_claims": PREFIX_CLAIMS},
        {"rulings": [
            {"claim_id": 1, "ruling": "supported", "evidence": "cargo build ran at tool 0"},
            {"claim_id": 2, "ruling": "unsupported", "evidence": "tool 0 shows error[E0433]"},
        ]},
    )
    v = evaluate(cfg(tmp_path), llm, "build it", result_with_buried_error())
    assert v.unsupported_claims == [PREFIX_CLAIMS[1]]   # the real one survives
    assert v.dropped_claims == [PREFIX_CLAIMS[0]]       # the false one is dropped
    assert v.adjudication_notes == []


def test_adjudication_anomalies_surface_on_the_verdict(tmp_path):
    """A malfunctioning adjudicator must not look like a well-behaved one."""
    llm = _critic_then_adjudicator(
        {"verdict": "revise", "score": 4, "critique": "c", "required_fixes": ["f"],
         "unsupported_claims": ["'x'", "'y'"]},
        {"rulings": [{"claim_id": 7, "ruling": "supported", "evidence": "e"}]},
    )
    v = evaluate(cfg(tmp_path), llm, "t", result_with_buried_error())
    assert v.unsupported_claims == ["'x'", "'y'"]       # nothing dropped on bad input
    assert any("out of range" in n for n in v.adjudication_notes)
    assert any("unchallenged" in n for n in v.adjudication_notes)
