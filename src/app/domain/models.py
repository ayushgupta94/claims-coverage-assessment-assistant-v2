"""Domain models -- the single source of truth for data shapes.

Persisted documents, tool inputs/outputs, the LLM's structured answer and the
final decision returned by the API are all defined here, so the contract
between ingestion, retrieval, tools, agent, decision policy and API is
explicit and validated everywhere.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------
# Product catalog vs. issued policy -- kept separate on purpose.
#
# PolicyProductVersion is the reusable product definition (e.g.
# "AUTO-GOLD-V1"): its rules and its policy wording are shared by every
# customer who bought that version. IssuedPolicy is one customer's purchase,
# carrying only per-customer values. A new wording becomes a new product
# version, so already-issued policies keep the wording they were sold under.
# --------------------------------------------------------------------------
class PolicyProductVersion(BaseModel):
    product_version_id: str
    product_id: str
    policy_type: str = Field(description="e.g. 'auto', 'health', 'home'")
    excluded_claim_types: list[str] = Field(default_factory=list)
    waiting_period_days: int = 0


class IssuedPolicy(BaseModel):
    policy_id: str
    customer_id: str
    product_version_id: str
    sum_insured: float
    inception_date: datetime


# --------------------------------------------------------------------------
# RAG corpus: produced by the ingestion pipeline from policy wording PDFs.
# A clause (child chunk) is what retrieval matches; its section (parent) is
# the broader context handed to the LLM alongside it.
# --------------------------------------------------------------------------
class PolicySection(BaseModel):
    section_id: str  # e.g. "AUTO-GOLD-V1-S4"
    product_version_id: str
    title: str  # e.g. "4. Exclusions"
    text: str
    page: int


class PolicyClause(BaseModel):
    clause_id: str  # e.g. "AUTO-GOLD-V1-4.2"
    product_version_id: str
    section_id: str
    title: str  # e.g. "4.2 Racing and Speed Testing"
    text: str
    page: int
    chunk_hash: str  # sha256 of the chunk content -> idempotent re-ingestion
    embedding: list[float] | None = None


class ClauseMatch(BaseModel):
    """A clause returned by retrieval. `score` is the fused (RRF) rank score;
    `vector_score` is the raw vector similarity (0-1) when the vector search
    found it, used as a retrieval-quality signal by the decision policy."""

    clause: PolicyClause
    score: float
    vector_score: float | None = None


class RetrievedClause(BaseModel):
    """A clause as returned by the retrieve_policy_clauses tool."""

    clause_id: str
    title: str
    page: int
    text: str
    section_id: str
    score: float
    vector_score: float | None


class RetrievedSection(BaseModel):
    section_id: str
    title: str
    text: str


# --------------------------------------------------------------------------
# Claims
# --------------------------------------------------------------------------
class ClaimStatus(StrEnum):
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class ClaimRequest(BaseModel):
    """A claim as submitted. "History" is not a separate entity: it is a query
    over the same collection for other claims on the same policy."""

    claim_id: str = Field(min_length=1, max_length=64)
    policy_id: str
    customer_id: str
    claim_type: str = Field(description="e.g. 'collision', 'theft', 'water_damage'")
    description: str = Field(min_length=1, max_length=4000)
    amount: float = Field(gt=0)
    incident_date: datetime
    filed_at: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------
# Deterministic tool outputs
# --------------------------------------------------------------------------
class CoverageRuleResult(BaseModel):
    is_covered: bool
    violated_rules: list[str] = Field(default_factory=list)
    notes: str = ""


class FraudRiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FraudRiskResult(BaseModel):
    risk_level: FraudRiskLevel
    risk_score: float = Field(ge=0.0, le=1.0)
    signals: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Agent: the LLM's structured final answer, and a record of one agent run
# --------------------------------------------------------------------------
class CoverageOutcome(StrEnum):
    COVERED = "covered"
    NOT_COVERED = "not_covered"
    PARTIALLY_COVERED = "partially_covered"


class AgentAnswer(BaseModel):
    """What the LLM must return (enforced with OpenAI Structured Outputs).
    It proposes an outcome, cites clauses and explains; code decides the
    final outcome, confidence and review routing (see decision/policy.py)."""

    claim_id: str
    coverage_outcome: CoverageOutcome
    supporting_clause_ids: list[str] = Field(description="clause_ids returned by retrieve_policy_clauses")
    reasoning: str = Field(description="2-4 sentences citing clause_ids")
    self_reported_confidence: float = Field(description="0-1; logged only, never returned")


class ToolStep(BaseModel):
    tool: str
    arguments: dict
    ok: bool
    latency_ms: int
    error: str | None = None


class TokenUsage(BaseModel):
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            llm_calls=self.llm_calls + other.llm_calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=round(self.cost_usd + other.cost_usd, 6),
        )


class AgentRun(BaseModel):
    """Everything one agent run produced -- the input to the decision policy."""

    answer: AgentAnswer | None  # None when the fallback path was used
    coverage: CoverageRuleResult
    fraud: FraudRiskResult
    retrieved: dict[str, RetrievedClause] = Field(default_factory=dict)  # clause_id -> clause
    steps: list[ToolStep] = Field(default_factory=list)
    guardrail_flags: list[str] = Field(default_factory=list)
    used_fallback: bool = False
    usage: TokenUsage = Field(default_factory=TokenUsage)


# --------------------------------------------------------------------------
# Final decision (the API contract) and human review
# --------------------------------------------------------------------------
class ReviewReason(StrEnum):
    NOT_COVERED = "not_covered"
    FRAUD_RISK = "fraud_risk_medium_or_high"
    LOW_CONFIDENCE = "low_confidence"
    LLM_CONTRADICTS_RULES = "llm_contradicts_rules"
    INJECTION_SUSPECTED = "injection_suspected"
    TOOL_ARGUMENT_VIOLATION = "tool_argument_violation"
    FALLBACK_USED = "fallback_used"


class Citation(BaseModel):
    clause_id: str
    title: str
    page: int
    text: str


class CoverageDecision(BaseModel):
    claim_id: str
    coverage_outcome: CoverageOutcome
    citations: list[Citation] = Field(default_factory=list)
    dropped_citations: list[str] = Field(default_factory=list, description="cited by the LLM but never retrieved")
    confidence_score: float = Field(ge=0.0, le=1.0)
    fraud_risk: FraudRiskResult
    coverage_rules: CoverageRuleResult
    requires_human_review: bool
    review_reasons: list[ReviewReason] = Field(default_factory=list)
    reasoning: str
    model: str
    prompt_version: str
    generated_at: datetime = Field(default_factory=_utcnow)


class HistoryEntry(BaseModel):
    """One audit-trail entry on a claim: who did what, when, and why."""

    at: datetime = Field(default_factory=_utcnow)
    actor: str  # "agent", "system" or the reviewer's name
    action: Literal["assessed", "auto_approved", "auto_rejected", "escalated", "approved", "overridden"]
    outcome: CoverageOutcome | None = None
    reason: str = ""
    model: str | None = None
    prompt_version: str | None = None


class ReviewRequest(BaseModel):
    action: Literal["approve", "override"]
    outcome: CoverageOutcome | None = Field(default=None, description="required when action=override")
    reason: str = Field(min_length=3, max_length=1000)
    reviewer: str = Field(min_length=1, max_length=100)


class ClaimRecord(BaseModel):
    """A claim with its lifecycle state, as stored and as returned by the API."""

    claim: ClaimRequest
    status: ClaimStatus
    decision: CoverageDecision | None = None
    steps: list[ToolStep] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    decision_history: list[HistoryEntry] = Field(default_factory=list)
