# ============================================================
# modules/w2ner_adapter.py
# W2NER Statement Entity Extractor
#
# Architecture: PhoBERT-base-v2 + BiLSTM + ConditionalLayerNorm
#               + Multi-granularity Dilated 2D-CNN
#               + Biaffine / MLP Co-Predictor
#
# Label set (2D grid):
#   NONE         - no relation between token pair
#   NNW          - Next-Neighboring-Word (sequential link)
#   THW-PAYER_ORG    - Tail-Head: paying organization entity
#   THW-SENDER_PER   - Tail-Head: individual payer entity
#   THW-INVOICE_ID   - Tail-Head: invoice reference number
#   THW-DISCOUNT_RATE- Tail-Head: discount percentage
#   THW-TAX_CODE     - Tail-Head: VAT tax code
#   THW-PERIOD       - Tail-Head: time period (month/year)
#
# Two execution modes:
#   1. REAL   — load .pt checkpoint, run PyTorch inference
#   2. DEMO   — HybridW2NERSimulator (regex + rule-based)
#               returns identical output schema including grid_2d_Relations
#               and entity_spans for Streamlit heatmap rendering.
# ============================================================

from __future__ import annotations

import json
import logging
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import FUZZY_MATCH_THRESHOLD

# ---------------------------------------------------------------------------
# Optional PyTorch / HuggingFace — graceful degradation
# ---------------------------------------------------------------------------
_HAS_TORCH = False
_HAS_TRANSFORMERS = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    logger.warning("[W2NER] PyTorch not available — DEMO mode only.")

try:
    from transformers import AutoModel, AutoTokenizer, BertConfig
    _HAS_TRANSFORMERS = True
except ImportError:
    logger.warning("[W2NER] Transformers not available — DEMO mode only.")


# ===========================================================================
# LABEL REGISTRY
# ===========================================================================

LABEL_NONE = "NONE"
LABEL_NNW  = "NNW"

# THW labels map to entity types
THW_LABELS: List[str] = [
    "THW-PAYER_ORG",
    "THW-SENDER_PER",
    "THW-INVOICE_ID",
    "THW-DISCOUNT_RATE",
    "THW-TAX_CODE",
    "THW-PERIOD",
]

ALL_LABELS: List[str] = [LABEL_NONE, LABEL_NNW] + THW_LABELS
LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(ALL_LABELS)}
ID2LABEL: Dict[int, str] = {i: l for l, i in LABEL2ID.items()}

NUM_LABELS = len(ALL_LABELS)   # 8

# Entity type names (stripped of THW- prefix)
ENTITY_TYPES: List[str] = [l.replace("THW-", "") for l in THW_LABELS]

# Default PhoBERT config
PHOBERT_MODEL_NAME = "vinai/phobert-base-v2"
W2NER_CHECKPOINT_ENV = "W2NER_CHECKPOINT_PATH"

# ===========================================================================
# OUTPUT DATA STRUCTURE
# ===========================================================================

@dataclass
class EntitySpan:
    """Single entity instance with token positions."""
    label:            str
    text:             str
    token_indices:    List[int]       # Token positions in the sentence
    is_discontinuous: bool = False    # True if span has gaps (W2NER specialty)
    is_nested:        bool = False    # True if overlaps with another entity
    char_start:       int  = -1
    char_end:         int  = -1

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class GridRelation:
    """One cell in the W2NER 2D word-word relation grid."""
    head_token_idx: int
    tail_token_idx: int
    head_token:     str
    tail_token:     str
    relation_type:  str    # NONE | NNW | THW-*
    confidence:     float  = 1.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class W2NEROutput:
    """Canonical output of W2NERStatementExtractor.extract_entities()."""
    raw_text:          str
    tokens:            List[str]
    entities: Dict[str, List] = field(default_factory=lambda: {
        "PAYER_ORG":     [],   # List[str]
        "SENDER_PER":    [],   # List[str]
        "INVOICE_ID":    [],   # List[str]  — normalized to 5-digit strings
        "DISCOUNT_RATE": [],   # List[float]
        "TAX_CODE":      [],   # List[str]
        "PERIOD":        [],   # List[str]
    })
    entity_spans:         List[Dict] = field(default_factory=list)
    grid_2d_Relations:    List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "raw_text":          self.raw_text,
            "tokens":            self.tokens,
            "entities":          self.entities,
            "entity_spans":      [s for s in self.entity_spans],
            "grid_2d_Relations": [r for r in self.grid_2d_Relations],
        }

    # Convenience accessors
    @property
    def payer_orgs(self)     -> List[str]:  return self.entities["PAYER_ORG"]
    @property
    def sender_persons(self) -> List[str]:  return self.entities["SENDER_PER"]
    @property
    def invoice_ids(self)    -> List[str]:  return self.entities["INVOICE_ID"]
    @property
    def discount_rates(self) -> List[float]:return self.entities["DISCOUNT_RATE"]
    @property
    def tax_codes(self)      -> List[str]:  return self.entities["TAX_CODE"]
    @property
    def periods(self)        -> List[str]:  return self.entities["PERIOD"]


# ===========================================================================
# PYTORCH MODEL ARCHITECTURE
# ===========================================================================

if _HAS_TORCH and _HAS_TRANSFORMERS:

    class ConditionalLayerNorm(nn.Module):
        """
        Layer normalization conditioned on an external vector (e.g. CLN in W2NER).
        gamma and beta are predicted from a condition vector h.
        """

        def __init__(self, hidden_size: int, cond_size: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.W_gamma = nn.Linear(cond_size, hidden_size, bias=False)
            self.W_beta  = nn.Linear(cond_size, hidden_size, bias=False)
            self.gamma   = nn.Parameter(torch.ones(hidden_size))
            self.beta    = nn.Parameter(torch.zeros(hidden_size))

        def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            # x: (B, N, H), cond: (B, cond_size)
            gamma = self.gamma + self.W_gamma(cond).unsqueeze(1)
            beta  = self.beta  + self.W_beta(cond).unsqueeze(1)
            mean  = x.mean(-1, keepdim=True)
            std   = x.std (-1, keepdim=True) + self.eps
            return gamma * (x - mean) / std + beta


    class DilatedConv2d(nn.Module):
        """
        Multi-granularity Dilated Convolutional Encoder for W2NER.
        Stacks convolutions with dilation rates [1, 2, 3] on the 2D token grid.
        """

        def __init__(self, in_channels: int, out_channels: int,
                     dilations: List[int] = None):
            super().__init__()
            if dilations is None:
                dilations = [1, 2, 3]
            self.convs = nn.ModuleList([
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=3, padding=d, dilation=d, bias=True,
                )
                for d in dilations
            ])
            self.norm = nn.LayerNorm(out_channels * len(dilations))
            self.act  = nn.GELU()
            self.proj = nn.Linear(out_channels * len(dilations), out_channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, C, N, N) — 2D word-pair grid
            outs = [conv(x) for conv in self.convs]  # each: (B, out_C, N, N)
            merged = torch.cat(outs, dim=1)           # (B, out_C*3, N, N)
            # Permute for LayerNorm: (B, N, N, out_C*3)
            merged = merged.permute(0, 2, 3, 1)
            merged = self.norm(merged)
            merged = self.act(merged)
            out    = self.proj(merged)                # (B, N, N, out_C)
            return out                                # (B, N, N, out_C)


    class Biaffine(nn.Module):
        """
        Biaffine scorer: score(i, j) = h_i^T W h_j + U h_i + V h_j + b
        Used as co-predictor alongside MLP in W2NER.
        """

        def __init__(self, n_in: int, n_out: int, bias_x: bool = True, bias_y: bool = True):
            super().__init__()
            self.n_in  = n_in
            self.n_out = n_out
            self.bias_x = bias_x
            self.bias_y = bias_y
            # Weight tensor: (n_out, n_in + bias_x, n_in + bias_y)
            self.weight = nn.Parameter(
                torch.zeros(n_out, n_in + int(bias_x), n_in + int(bias_y))
            )
            nn.init.orthogonal_(self.weight)

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # x, y: (B, N, n_in)
            if self.bias_x:
                x = torch.cat([x, x.new_ones(*x.shape[:-1], 1)], -1)
            if self.bias_y:
                y = torch.cat([y, y.new_ones(*y.shape[:-1], 1)], -1)
            # (B, N, n_out, N) = einsum B N d_x, n_out d_x d_y, B N d_y
            scores = torch.einsum("bnd,odc,bmc->bnom", x, self.weight, y)
            return scores.permute(0, 1, 3, 2)   # (B, N, N, n_out)


    class W2NERModel(nn.Module):
        """
        Full W2NER architecture (compact version for inference):
          PhoBERT encoder -> BiLSTM -> ConditionalLayerNorm ->
          DilatedConv2D (dilations 1,2,3) -> Biaffine + MLP co-predictor ->
          Softmax over label set
        """

        def __init__(
            self,
            encoder_name:   str = PHOBERT_MODEL_NAME,
            hidden_size:    int = 768,
            lstm_hidden:    int = 384,
            cnn_channels:   int = 128,
            num_labels:     int = NUM_LABELS,
            dropout:        float = 0.2,
            dilations:      List[int] = None,
        ):
            super().__init__()
            if dilations is None:
                dilations = [1, 2, 3]

            # 1. PhoBERT encoder
            self.encoder = AutoModel.from_pretrained(encoder_name, add_pooling_layer=False)
            encoder_dim  = self.encoder.config.hidden_size

            # 2. BiLSTM context encoder
            self.bilstm = nn.LSTM(
                input_size    = encoder_dim,
                hidden_size   = lstm_hidden,
                num_layers    = 2,
                batch_first   = True,
                bidirectional = True,
                dropout       = dropout,
            )
            bilstm_out = lstm_hidden * 2   # 768

            # 3. Conditional Layer Normalization
            self.cln = ConditionalLayerNorm(
                hidden_size = bilstm_out,
                cond_size   = bilstm_out,
            )

            # 4. Dilated 2D CNN on the outer-product grid
            # Grid input channels = bilstm_out (concatenated head+tail features -> bilstm_out)
            self.conv2d = DilatedConv2d(bilstm_out, cnn_channels, dilations=dilations)

            # 5. Biaffine co-predictor
            self.biaffine = Biaffine(bilstm_out // 2, num_labels)

            # 6. MLP co-predictor (on CNN output)
            self.mlp = nn.Sequential(
                nn.Linear(cnn_channels, cnn_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(cnn_channels, num_labels),
            )

            # 7. Dropout
            self.dropout = nn.Dropout(dropout)

        def encode_tokens(
            self,
            input_ids:      torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Run PhoBERT + BiLSTM, return (B, N, bilstm_out)."""
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            seq_out = outputs.last_hidden_state   # (B, L, 768)
            seq_out = self.dropout(seq_out)
            lstm_out, _ = self.bilstm(seq_out)    # (B, L, 768)
            return self.dropout(lstm_out)

        def forward(
            self,
            input_ids:      torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Returns:
                logits: (B, N, N, num_labels) — the 2D word-pair label grid.
            """
            B, L = input_ids.shape
            h = self.encode_tokens(input_ids, attention_mask, token_type_ids)
            # N = sequence length (excluding [CLS] / [SEP] for real use)
            N = L

            # -- Build 2D grid feature map --
            # Outer product of token representations: (B, N, N, bilstm_out)
            h_i = h.unsqueeze(2).expand(B, N, N, -1)  # head
            h_j = h.unsqueeze(1).expand(B, N, N, -1)  # tail
            # Fuse head+tail via CLN: condition head by tail
            cond = h_j.reshape(B * N, N, -1)
            base = h_i.reshape(B * N, N, -1)
            fused = self.cln(base, cond.mean(dim=1))   # (B*N, N, bilstm_out)
            fused = fused.reshape(B, N, N, -1)         # (B, N, N, bilstm_out)

            # -- CNN branch --
            cnn_in = fused.permute(0, 3, 1, 2)         # (B, bilstm_out, N, N)
            cnn_out = self.conv2d(cnn_in)               # (B, N, N, cnn_channels)
            mlp_logits = self.mlp(cnn_out)              # (B, N, N, num_labels)

            # -- Biaffine branch (operates on half-dim projections) --
            d = h.shape[-1] // 2
            biaffine_logits = self.biaffine(h[..., :d], h[..., d:])  # (B, N, N, num_labels)

            # -- Combine --
            logits = mlp_logits + biaffine_logits       # (B, N, N, num_labels)
            return logits


# ===========================================================================
# W2NER GRID DECODER
# ===========================================================================

def decode_2d_grid(
    grid_logits: "torch.Tensor",
    tokens:      List[str],
    id2label:    Dict[int, str] = None,
    threshold:   float = 0.5,
) -> Tuple[List[EntitySpan], List[GridRelation]]:
    """
    Decode the W2NER 2D word-pair relation grid into entity spans.

    Algorithm:
      1. Convert logits -> argmax label IDs for each (i, j) pair.
      2. Build directed graph: i -> j with label type.
      3. For each THW-* arc (i, j): entity ENDS at j and STARTS somewhere.
         Trace backward along NNW arcs from j to find the full span.
      4. Collect discontinuous spans by tracking all NNW chains.

    Args:
        grid_logits: Tensor (N, N, num_labels) or (1, N, N, num_labels).
        tokens:      List of N token strings.
        id2label:    Map from label ID -> label string.
        threshold:   Minimum softmax probability to accept a non-NONE label.

    Returns:
        (entity_spans, grid_relations)
    """
    if id2label is None:
        id2label = ID2LABEL

    if _HAS_TORCH:
        if grid_logits.dim() == 4:
            grid_logits = grid_logits.squeeze(0)   # (N, N, num_labels)
        probs  = F.softmax(grid_logits, dim=-1)    # (N, N, num_labels)
        pred   = grid_logits.argmax(dim=-1)        # (N, N)
        prob_v = probs.max(dim=-1).values          # (N, N)
    else:
        raise RuntimeError("PyTorch required for decode_2d_grid with real tensors.")

    N = len(tokens)
    entity_spans:  List[EntitySpan]  = []
    grid_relations: List[GridRelation] = []

    # Build adjacency: adj[i][j] = label_str
    adj: Dict[int, Dict[int, str]] = {i: {} for i in range(N)}
    for i in range(N):
        for j in range(N):
            lbl_id = int(pred[i, j].item())
            lbl    = id2label.get(lbl_id, LABEL_NONE)
            conf   = float(prob_v[i, j].item())
            if lbl != LABEL_NONE and conf >= threshold:
                adj[i][j] = lbl
                grid_relations.append(GridRelation(
                    head_token_idx = i,
                    tail_token_idx = j,
                    head_token     = tokens[i] if i < N else "",
                    tail_token     = tokens[j] if j < N else "",
                    relation_type  = lbl,
                    confidence     = conf,
                ))

    # Identify all THW arcs -> entity boundaries
    thw_arcs: List[Tuple[int, int, str]] = []   # (head_i, tail_j, entity_type)
    for i in range(N):
        for j, lbl in adj[i].items():
            if lbl.startswith("THW-"):
                thw_arcs.append((i, j, lbl.replace("THW-", "")))

    # For each THW arc, trace NNW chain backward from head_i to find full span
    def trace_nnw_chain(start: int) -> List[int]:
        """Trace NNW arcs forward from start to collect all tokens in span."""
        chain = [start]
        visited = {start}
        cur = start
        for _ in range(N):
            nexts = [j for j, l in adj[cur].items() if l == LABEL_NNW and j not in visited]
            if not nexts:
                break
            cur = nexts[0]
            chain.append(cur)
            visited.add(cur)
        return chain

    # Track entity occurrences for nested detection
    all_spans_indices: List[Tuple[int, ...]] = []

    for head_i, tail_j, etype in thw_arcs:
        chain = trace_nnw_chain(head_i)
        # Include tail_j if not already in chain
        if tail_j not in chain:
            chain.append(tail_j)
        chain = sorted(set(chain))

        # Check discontinuous: gaps in token indices
        is_disc = any(chain[k + 1] - chain[k] > 1 for k in range(len(chain) - 1))

        # Check nested: overlaps with existing spans
        chain_set = set(chain)
        is_nested = any(chain_set & set(prev) and chain_set != set(prev)
                        for prev in all_spans_indices)

        all_spans_indices.append(tuple(chain))

        span_text = " ".join(tokens[idx] for idx in chain if idx < N)
        entity_spans.append(EntitySpan(
            label            = etype,
            text             = span_text,
            token_indices    = chain,
            is_discontinuous = is_disc,
            is_nested        = is_nested,
        ))

    return entity_spans, grid_relations


# ===========================================================================
# HYBRID SIMULATOR (DEMO MODE)
# ===========================================================================

class HybridW2NERSimulator:
    """
    Rule-based + regex simulator that produces W2NER-compatible output
    (including grid_2d_Relations and entity_spans) without requiring a GPU
    or downloaded model weights.

    Covers all entity types used in Vietnamese bank statement NER:
      PAYER_ORG, SENDER_PER, INVOICE_ID, DISCOUNT_RATE, TAX_CODE, PERIOD.

    Special handling for W2NER's hallmark capabilities:
      - Discontinuous entities: "HD SO 00102 VA 00105" -> two INVOICE_IDs
      - Nested entities: org name containing a person's name
      - Multi-token entities: full company name spans
    """

    # --- Regex patterns ---

    # Invoice ID: 5-digit number, possibly preceded by "HD", "SO", "HDxx"
    _RE_INVOICE = re.compile(
        r"(?:HD\s*(?:SO\s*)?)?(\d{5})\b", re.IGNORECASE
    )
    # Also match "00102" standalone near keywords
    _RE_INVOICE_NEAR = re.compile(
        r"\b(\d{4,6})\b"
    )

    # Discount rate: "2%", "2 %", "CK 2%", "CHIET KHAU 3%"
    _RE_DISCOUNT = re.compile(
        r"(?:CK|CHIET\s*KHAU|TRU\s*CK|DISCOUNT)?\s*(\d{1,2}(?:[.,]\d+)?)\s*(?:PC|%)",
        re.IGNORECASE,
    )

    # Tax code: 10–13 digit number after MST/TAX keyword or standalone pattern
    _RE_TAX_CODE = re.compile(
        r"(?:MST|MA\s*SO\s*THUE|TAX\s*CODE)[:\s]*(\d{10,13})",
        re.IGNORECASE,
    )

    # Period: "THANG 9", "THANG 09/2024", "T9", "T10", "QUY 3"
    _RE_PERIOD = re.compile(
        r"\b(?:THANG\s*(\d{1,2}(?:[/\-]\d{4})?)|T(\d{1,2})\b|QUY\s*(\d{1,2}))",
        re.IGNORECASE,
    )

    # Individual sender (personal name) keywords
    _SENDER_KEYWORDS = [
        "CHUYEN KHOAN", "CHUYEN TIEN", "THANH TOAN CONG NO",
        "TT CONG NO", "NOP TIEN", "CK TIEN",
    ]

    # Organization keywords
    _ORG_PREFIXES = [
        "CTY TNHH", "CONG TY TNHH", "CONG TY CO PHAN", "CTY CP",
        "DN TN", "DOANH NGHIEP TU NHAN", "CONG TY", "CTY",
        "TONG CONG TY", "TAP DOAN", "DUOC PHAM",
    ]

    # Vietnamese person name patterns (2–4 words, all UPPER or Title case)
    _RE_PERSON = re.compile(
        r"\b([A-Z][A-Z]+(?:\s+[A-Z][A-Z]+){1,3})\s+"
        r"(?:CHUYEN\s*KHOAN|THANH\s*TOAN|CHUYEN\s*TIEN|TT\s*CONG\s*NO|NOP\s*TIEN)",
        re.IGNORECASE,
    )

    def __init__(self):
        # Build org extraction regex from prefixes
        prefix_pat = "|".join(re.escape(p) for p in self._ORG_PREFIXES)
        self._RE_ORG = re.compile(
            rf"(?:{prefix_pat})\s+([A-Z\s\u00C0-\u024F]+?)(?:\s+(?:THANH\s*TOAN|TT|CHUYEN|HD|TAX|MST|\d)|\Z)",
            re.IGNORECASE | re.UNICODE,
        )

    def normalize_text(self, text: str) -> str:
        """Remove accents for regex matching."""
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c)).upper()

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenizer preserving punctuation as separate tokens."""
        tokens = []
        for tok in re.split(r"\s+", text.strip()):
            # Further split on certain punctuation but keep the char
            subtoks = re.findall(r"[^\s,;]+|[,;]", tok)
            tokens.extend(subtoks)
        return [t for t in tokens if t]

    def find_token_span(
        self, tokens: List[str], phrase: str
    ) -> Optional[List[int]]:
        """Find the token indices of phrase in token list (case-insensitive)."""
        phrase_tokens = self.tokenize(phrase)
        n, m = len(tokens), len(phrase_tokens)
        for start in range(n - m + 1):
            if all(
                tokens[start + k].upper() == phrase_tokens[k].upper()
                for k in range(m)
            ):
                return list(range(start, start + m))
        # Fuzzy: partial match
        phrase_norm = self.normalize_text(phrase)
        for start in range(n - m + 1):
            window = " ".join(tokens[start:start + m]).upper()
            if phrase_norm in window or window in phrase_norm:
                return list(range(start, start + m))
        return None

    def extract(self, raw_text: str) -> W2NEROutput:
        """Main extraction entry point."""
        norm = self.normalize_text(raw_text)
        tokens = self.tokenize(raw_text)
        N = len(tokens)

        output = W2NEROutput(raw_text=raw_text, tokens=tokens)
        entity_spans:   List[Dict] = []
        grid_relations: List[Dict] = []

        # Track all discovered spans for nested detection
        used_indices: List[Tuple[int, ...]] = []

        # -------------------------------------------------------------------
        # 1. EXTRACT INVOICE IDs (Discontinuous entity specialty)
        # -------------------------------------------------------------------
        invoice_ids_found: List[str] = []
        invoice_id_spans:  List[List[int]] = []

        # Primary: near "HD" keyword
        for m in self._RE_INVOICE.finditer(norm):
            inv_num = m.group(1).zfill(5)
            if inv_num not in invoice_ids_found:
                invoice_ids_found.append(inv_num)
                # Find token span
                span = self._locate_number_span(tokens, m.group(1))
                invoice_id_spans.append(span)

        # Secondary: standalone 5-digit numbers near TT/HD keywords
        # Only if no primary matches
        if not invoice_ids_found:
            kw_positions = []
            for idx, tok in enumerate(tokens):
                if tok.upper() in {"HD", "SO", "TT", "THANH", "TOAN"}:
                    kw_positions.append(idx)
            for m in self._RE_INVOICE_NEAR.finditer(norm):
                num_str = m.group(1).zfill(5)
                if 4 <= len(m.group(1)) <= 6 and num_str not in invoice_ids_found:
                    # Check if near keyword
                    num_span = self._locate_number_span(tokens, m.group(1))
                    if any(abs(num_span[0] - kp) <= 4 for kp in kw_positions):
                        invoice_ids_found.append(num_str)
                        invoice_id_spans.append(num_span)

        # Normalize to 5-digit strings
        output.entities["INVOICE_ID"] = [str(inv).zfill(5) for inv in invoice_ids_found]

        # Build entity spans + grid relations for invoice IDs
        # Multiple IDs form a DISCONTINUOUS entity if extracted from one statement
        if len(invoice_ids_found) > 1:
            # Discontinuous: all IDs linked by NNW arcs with final THW-INVOICE_ID
            all_idx = []
            for sp in invoice_id_spans:
                all_idx.extend(sp)
            all_idx = sorted(set(all_idx))

            # Check gaps -> discontinuous
            is_disc = any(
                all_idx[k + 1] - all_idx[k] > 1
                for k in range(len(all_idx) - 1)
            )
            span_text = " | ".join(
                str(inv) for inv in invoice_ids_found
            )
            entity_spans.append(EntitySpan(
                label            = "INVOICE_ID",
                text             = span_text,
                token_indices    = all_idx,
                is_discontinuous = is_disc,
                is_nested        = False,
            ).to_dict())

            # Build NNW arcs + THW arc
            if len(all_idx) >= 2:
                for k in range(len(all_idx) - 1):
                    grid_relations.append(GridRelation(
                        head_token_idx = all_idx[k],
                        tail_token_idx = all_idx[k + 1],
                        head_token = tokens[all_idx[k]] if all_idx[k] < N else "",
                        tail_token = tokens[all_idx[k + 1]] if all_idx[k + 1] < N else "",
                        relation_type  = LABEL_NNW,
                        confidence     = 0.95,
                    ).to_dict())
                # THW arc: last-token -> first-token marks the entity
                grid_relations.append(GridRelation(
                    head_token_idx = all_idx[-1],
                    tail_token_idx = all_idx[0],
                    head_token = tokens[all_idx[-1]] if all_idx[-1] < N else "",
                    tail_token = tokens[all_idx[0]] if all_idx[0] < N else "",
                    relation_type  = "THW-INVOICE_ID",
                    confidence     = 0.97,
                ).to_dict())
            used_indices.extend(invoice_id_spans)

        elif len(invoice_ids_found) == 1:
            sp = invoice_id_spans[0]
            entity_spans.append(EntitySpan(
                label         = "INVOICE_ID",
                text          = invoice_ids_found[0],
                token_indices = sp,
            ).to_dict())
            self._add_flat_span_relations(grid_relations, sp, "THW-INVOICE_ID", tokens)
            used_indices.append(tuple(sp))

        # -------------------------------------------------------------------
        # 2. EXTRACT DISCOUNT RATE
        # -------------------------------------------------------------------
        for m in self._RE_DISCOUNT.finditer(norm):
            rate_str = m.group(1).replace(",", ".")
            try:
                rate_pct = float(rate_str)
                rate_frac = rate_pct / 100.0
                output.entities["DISCOUNT_RATE"].append(round(rate_frac, 4))
                rate_span = self._locate_number_span(tokens, rate_str.rstrip("0").rstrip("."))
                entity_spans.append(EntitySpan(
                    label         = "DISCOUNT_RATE",
                    text          = f"{rate_pct}%",
                    token_indices = rate_span,
                ).to_dict())
                self._add_flat_span_relations(grid_relations, rate_span, "THW-DISCOUNT_RATE", tokens)
            except ValueError:
                pass

        # -------------------------------------------------------------------
        # 3. EXTRACT TAX CODE
        # -------------------------------------------------------------------
        for m in self._RE_TAX_CODE.finditer(norm):
            tc = m.group(1)
            output.entities["TAX_CODE"].append(tc)
            tc_span = self._locate_number_span(tokens, tc)
            entity_spans.append(EntitySpan(
                label         = "TAX_CODE",
                text          = tc,
                token_indices = tc_span,
            ).to_dict())
            self._add_flat_span_relations(grid_relations, tc_span, "THW-TAX_CODE", tokens)

        # -------------------------------------------------------------------
        # 4. EXTRACT TIME PERIOD
        # -------------------------------------------------------------------
        for m in self._RE_PERIOD.finditer(norm):
            if m.group(1):
                period_str = f"THANG {m.group(1)}"
            elif m.group(2):
                period_str = f"T{m.group(2)}"
            else:
                period_str = f"QUY {m.group(3)}"
            output.entities["PERIOD"].append(period_str)
            # Find span
            kw = m.group(0).split()[0]
            p_span = self._locate_keyword_span(tokens, kw)
            entity_spans.append(EntitySpan(
                label         = "PERIOD",
                text          = period_str,
                token_indices = p_span,
            ).to_dict())
            self._add_flat_span_relations(grid_relations, p_span, "THW-PERIOD", tokens)

        # -------------------------------------------------------------------
        # 5. EXTRACT INDIVIDUAL SENDER (SENDER_PER) OR UNPREFIXED ORG
        # -------------------------------------------------------------------
        NON_PERSON_ORG_INDICATORS = [
            "CTY", "CONG TY", "TNHH", "CP", "PHARMA", "DUOC PHAM", "XNK",
            "VAN TAI", "XAY DUNG", "THUONG MAI", "THUC PHAM", "NOI THAT",
            "KY THUAT", "DOANH NGHIEP", "ENTERPRISE", "CORP", "JSC"
        ]

        for m in self._RE_PERSON.finditer(raw_text.upper()):
            name = m.group(1).strip()
            words = name.split()

            # Check if this is actually an organization without standard prefix
            is_org_name = any(ind in name.upper() for ind in NON_PERSON_ORG_INDICATORS)
            if is_org_name:
                if name not in output.entities["PAYER_ORG"]:
                    output.entities["PAYER_ORG"].append(name)
                    org_span = self.find_token_span(tokens, name) or [0]
                    entity_spans.append(EntitySpan(
                        label         = "PAYER_ORG",
                        text          = name,
                        token_indices = org_span,
                    ).to_dict())
                    self._add_flat_span_relations(grid_relations, org_span, "THW-PAYER_ORG", tokens)
                    used_indices.append(tuple(org_span))
                continue

            # Validate real personal name: 2-4 words, each >= 2 chars (not just initials)
            if 2 <= len(words) <= 5 and all(len(w) >= 2 for w in words):
                if name not in output.entities["SENDER_PER"]:
                    output.entities["SENDER_PER"].append(name)
                    p_span = self.find_token_span(tokens, name) or [0]
                    entity_spans.append(EntitySpan(
                        label         = "SENDER_PER",
                        text          = name,
                        token_indices = p_span,
                    ).to_dict())
                    self._add_flat_span_relations(grid_relations, p_span, "THW-SENDER_PER", tokens)

        # -------------------------------------------------------------------
        # 6. EXTRACT PAYER ORG (PAYER_ORG)
        # -------------------------------------------------------------------
        # Strategy A: Explicit prefix match
        org_found = False
        for m in self._RE_ORG.finditer(norm):
            # Reconstruct with prefix
            full_match_start = m.start()
            # Find the prefix token
            prefix_end = m.start(1)
            org_name_norm = m.group(1).strip()
            # Map back to original text approximate
            org_name = self._remap_to_original(raw_text, org_name_norm)
            if org_name and len(org_name) >= 4:
                # Find full org (prefix + name)
                prefix_region = norm[:prefix_end].rsplit(None, 1)
                prefix = prefix_region[-1] if prefix_region else ""
                full_org = f"{prefix} {org_name}".strip()
                if full_org not in output.entities["PAYER_ORG"]:
                    output.entities["PAYER_ORG"].append(full_org)
                    org_span = self.find_token_span(tokens, full_org) or list(range(min(4, N)))
                    is_nested = any(set(org_span) & set(ui) for ui in used_indices)
                    entity_spans.append(EntitySpan(
                        label         = "PAYER_ORG",
                        text          = full_org,
                        token_indices = org_span,
                        is_nested     = is_nested,
                    ).to_dict())
                    self._add_flat_span_relations(grid_relations, org_span, "THW-PAYER_ORG", tokens)
                    org_found = True
                    used_indices.append(tuple(org_span))

        # Strategy B: If no SENDER_PER and no ORG prefix found,
        # try treating the first N tokens as org name
        if not org_found and not output.entities["SENDER_PER"]:
            # Heuristic: take first 3-6 tokens that look like a company name
            candidate_toks = tokens[:6]
            candidate_name = " ".join(candidate_toks)
            # Only if no digit-only tokens
            if all(not tok.isdigit() for tok in candidate_toks) and len(candidate_toks) >= 2:
                output.entities["PAYER_ORG"].append(candidate_name)
                org_span = list(range(len(candidate_toks)))
                entity_spans.append(EntitySpan(
                    label         = "PAYER_ORG",
                    text          = candidate_name,
                    token_indices = org_span,
                ).to_dict())
                self._add_flat_span_relations(grid_relations, org_span, "THW-PAYER_ORG", tokens)

        output.entity_spans      = entity_spans
        output.grid_2d_Relations = grid_relations
        return output

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _locate_number_span(self, tokens: List[str], number_str: str) -> List[int]:
        """Find token index(es) where number_str appears."""
        clean = re.sub(r"[^\d]", "", str(number_str))
        for i, tok in enumerate(tokens):
            tok_clean = re.sub(r"[^\d]", "", tok)
            if tok_clean == clean:
                return [i]
        # Not found — return last token as placeholder
        return [len(tokens) - 1] if tokens else [0]

    def _locate_keyword_span(self, tokens: List[str], keyword: str) -> List[int]:
        """Find first token matching keyword (case-insensitive)."""
        kw_up = keyword.upper()
        for i, tok in enumerate(tokens):
            if tok.upper() == kw_up or kw_up in tok.upper():
                return [i, i + 1] if i + 1 < len(tokens) else [i]
        return [0]

    def _add_flat_span_relations(
        self,
        relations: List[Dict],
        span: List[int],
        thw_label: str,
        tokens: List[str],
    ) -> None:
        """
        Add NNW arcs for consecutive token pairs + THW arc from last to first
        to encode a flat (non-discontinuous) entity in the W2NER grid.
        """
        N = len(tokens)
        if not span:
            return
        # NNW arcs
        for k in range(len(span) - 1):
            i, j = span[k], span[k + 1]
            relations.append(GridRelation(
                head_token_idx = i,
                tail_token_idx = j,
                head_token = tokens[i] if i < N else "",
                tail_token = tokens[j] if j < N else "",
                relation_type  = LABEL_NNW,
                confidence     = 0.93,
            ).to_dict())
        # THW arc (tail -> head to close the entity)
        relations.append(GridRelation(
            head_token_idx = span[-1],
            tail_token_idx = span[0],
            head_token = tokens[span[-1]] if span[-1] < N else "",
            tail_token = tokens[span[0]] if span[0] < N else "",
            relation_type  = thw_label,
            confidence     = 0.95,
        ).to_dict())

    def _remap_to_original(self, original: str, norm_text: str) -> str:
        """
        Attempt to find the accent-stripped norm_text substring
        in the original Vietnamese text and return the original version.
        """
        orig_norm = unicodedata.normalize("NFKD", original)
        orig_ascii = "".join(c for c in orig_norm if not unicodedata.combining(c)).upper()
        idx = orig_ascii.find(norm_text.upper())
        if idx >= 0:
            return original[idx: idx + len(norm_text)].strip()
        return norm_text   # fallback: return normalized version


# ===========================================================================
# CHECKPOINT LOADER
# ===========================================================================

def load_pretrained_w2ner(
    checkpoint_path: str,
    device: str = "cpu",
    encoder_name: str = PHOBERT_MODEL_NAME,
) -> Optional["W2NERModel"]:
    """
    Load a W2NER model from a .pt checkpoint file.

    Args:
        checkpoint_path: Path to the .pt state dict file.
        device:          "cpu" or "cuda".
        encoder_name:    HuggingFace model name for the encoder.

    Returns:
        W2NERModel instance or None if unavailable.
    """
    if not _HAS_TORCH or not _HAS_TRANSFORMERS:
        logger.warning("[W2NER] PyTorch/Transformers not available. Cannot load checkpoint.")
        return None

    path = Path(checkpoint_path)
    if not path.exists():
        logger.warning(f"[W2NER] Checkpoint not found: {checkpoint_path}")
        return None

    try:
        model = W2NERModel(encoder_name=encoder_name)
        state = torch.load(str(path), map_location=device)
        # Support both raw state_dict and wrapped dicts
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        logger.info(f"[W2NER] Loaded checkpoint: {checkpoint_path} on {device}")
        return model
    except Exception as exc:
        logger.error(f"[W2NER] Failed to load checkpoint: {exc}")
        return None


# ===========================================================================
# MAIN CLASS: W2NERStatementExtractor
# ===========================================================================

class W2NERStatementExtractor:
    """
    Vietnamese bank statement NER extractor using W2NER architecture.

    Execution modes:
      - REAL:  Loads PhoBERT + W2NER from .pt checkpoint for full inference.
      - DEMO:  HybridW2NERSimulator — deterministic rule-based extraction
               with identical output schema (including 2D grid relations).

    Usage:
        extractor = W2NERStatementExtractor()
        result = extractor.extract_entities("CTY HOANG GIA TT HD SO 00113 00114")
        print(result.invoice_ids)   # ['00113', '00114']
        print(result.payer_orgs)    # ['CTY HOANG GIA']
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device:          str = "cpu",
        encoder_name:    str = PHOBERT_MODEL_NAME,
        max_seq_len:     int = 128,
    ):
        self.device       = device
        self.max_seq_len  = max_seq_len
        self.encoder_name = encoder_name
        self._mode        = "DEMO"
        self._model: Optional["W2NERModel"] = None
        self._tokenizer = None
        self._simulator = HybridW2NERSimulator()

        # Determine checkpoint path
        cp = checkpoint_path or os.environ.get(W2NER_CHECKPOINT_ENV, "")
        if cp:
            model = load_pretrained_w2ner(cp, device=device, encoder_name=encoder_name)
            if model is not None:
                self._model = model
                self._mode  = "REAL"
                if _HAS_TRANSFORMERS:
                    self._tokenizer = AutoTokenizer.from_pretrained(encoder_name)
                    logger.info(f"[W2NER] REAL mode — tokenizer: {encoder_name}")

        logger.info(f"[W2NER] W2NERStatementExtractor initialized in {self._mode} mode.")

    @property
    def mode(self) -> str:
        return self._mode

    # -----------------------------------------------------------------------
    # PRIMARY API
    # -----------------------------------------------------------------------

    def extract_entities(self, raw_text: str) -> W2NEROutput:
        """
        Extract all named entities from a raw bank statement description string.

        Args:
            raw_text: Raw Vietnamese bank transaction description.
                      E.g. "CTY HOANG GIA TT HD SO 00113 THANG 9 VA 00114 TRU CK 2PC"

        Returns:
            W2NEROutput with fields:
              .raw_text, .tokens, .entities{PAYER_ORG, SENDER_PER, INVOICE_ID,
              DISCOUNT_RATE, TAX_CODE, PERIOD}, .entity_spans, .grid_2d_Relations
        """
        if self._mode == "REAL":
            return self._extract_real(raw_text)
        else:
            return self._extract_demo(raw_text)

    def extract_batch(self, texts: List[str]) -> List[W2NEROutput]:
        """Extract entities from a list of statements."""
        return [self.extract_entities(t) for t in texts]

    def to_w2ner_json(self, raw_text: str) -> str:
        """Return JSON string of extraction result for storage in DB w2ner_json column."""
        result = self.extract_entities(raw_text)
        return json.dumps(result.to_dict(), ensure_ascii=False, indent=None)

    # -----------------------------------------------------------------------
    # REAL INFERENCE
    # -----------------------------------------------------------------------

    def _extract_real(self, raw_text: str) -> W2NEROutput:
        """Full PyTorch W2NER inference pipeline."""
        assert self._model is not None and self._tokenizer is not None

        # Tokenize
        enc = self._tokenizer(
            raw_text,
            max_length  = self.max_seq_len,
            truncation  = True,
            return_tensors = "pt",
            padding     = "max_length",
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        token_type_ids = enc.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        with torch.no_grad():
            logits = self._model(input_ids, attention_mask, token_type_ids)
            # logits: (1, L, L, num_labels)

        # Get actual tokens (strip [CLS], [SEP], padding)
        tokens_ids = input_ids[0].tolist()
        tokens_str = self._tokenizer.convert_ids_to_tokens(tokens_ids)
        # Remove special tokens
        real_tokens = [
            t for t in tokens_str
            if t not in {
                self._tokenizer.cls_token,
                self._tokenizer.sep_token,
                self._tokenizer.pad_token,
            }
        ]
        # Trim logits to real length
        L_real = len(real_tokens)
        logits_trimmed = logits[0, 1:L_real + 1, 1:L_real + 1, :]

        # Decode
        entity_spans, grid_relations = decode_2d_grid(
            logits_trimmed, real_tokens, ID2LABEL
        )

        # Build output
        output = W2NEROutput(raw_text=raw_text, tokens=real_tokens)
        for span in entity_spans:
            etype = span.label
            if etype in output.entities:
                val = span.text
                # Normalize special types
                if etype == "INVOICE_ID":
                    val = re.sub(r"\D", "", val).zfill(5)
                elif etype == "DISCOUNT_RATE":
                    try:
                        pct = float(re.sub(r"[^\d.]", "", val))
                        val = round(pct / 100.0, 4) if pct > 1 else round(pct, 4)
                    except ValueError:
                        val = 0.0
                output.entities[etype].append(val)

        output.entity_spans      = [s.to_dict() for s in entity_spans]
        output.grid_2d_Relations = [r.to_dict() for r in grid_relations]
        return output

    # -----------------------------------------------------------------------
    # DEMO INFERENCE
    # -----------------------------------------------------------------------

    def _extract_demo(self, raw_text: str) -> W2NEROutput:
        """HybridW2NERSimulator-based extraction (no GPU needed)."""
        return self._simulator.extract(raw_text)

    # -----------------------------------------------------------------------
    # GRID VISUALIZATION HELPER (for Streamlit heatmap)
    # -----------------------------------------------------------------------

    def build_grid_matrix(
        self,
        output:   W2NEROutput,
        label_id: Optional[str] = None,
    ) -> Tuple[List[List[float]], List[str], List[str]]:
        """
        Convert grid_2d_Relations into a 2D matrix for heatmap rendering.

        Args:
            output:   W2NEROutput from extract_entities().
            label_id: If set, return binary matrix for one label; else argmax.

        Returns:
            (matrix, row_labels, col_labels)
            matrix: N x N float values (confidence or 0/1)
            row_labels: token strings (head axis)
            col_labels: token strings (tail axis)
        """
        tokens = output.tokens
        N = len(tokens)
        matrix = [[0.0] * N for _ in range(N)]

        for rel in output.grid_2d_Relations:
            i = rel.get("head_token_idx", 0)
            j = rel.get("tail_token_idx", 0)
            lbl = rel.get("relation_type", LABEL_NONE)
            conf = float(rel.get("confidence", 1.0))
            if 0 <= i < N and 0 <= j < N:
                if label_id is None:
                    # Use label ID as intensity (non-NONE = colored)
                    if lbl != LABEL_NONE:
                        matrix[i][j] = max(matrix[i][j], conf)
                else:
                    if lbl == label_id:
                        matrix[i][j] = conf

        return matrix, tokens, tokens

    # -----------------------------------------------------------------------
    # REPR
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"W2NERStatementExtractor(mode='{self._mode}', device='{self.device}')"


# ===========================================================================
# CONVENIENCE FUNCTIONS
# ===========================================================================

def quick_extract(raw_text: str) -> Dict[str, Any]:
    """
    One-shot extraction without instantiating the class.
    Always runs in DEMO mode. Returns dict.
    """
    extractor = W2NERStatementExtractor()
    return extractor.extract_entities(raw_text).to_dict()


def demo_showcase() -> None:
    """
    Print a full demo of W2NER extraction on 5 representative statement strings
    covering all entity types and special cases (discontinuous, nested).
    """
    test_cases = [
        # A1: Flat entity, single invoice ID
        "CTY TNHH TM MINH PHAT TT TIEN HANG HD 00101",
        # A2: Multi-invoice (discontinuous INVOICE_ID) + discount rate
        "CTY CP XD AN BINH THANH TOAN HD SO 00102 THANG 9 VA HD 00105 TRU CK 2PC",
        # B: Subset sum — no invoice ID, only org
        "CTY HOANG GIA TT TIEN HANG DOT 1 THANG 10",
        # D: Individual sender (SENDER_PER) — no org
        "LE VAN TUNG CHUYEN KHOAN TIEN HANG THANG 11",
        # D2: Another individual
        "NGUYEN THI MAI TT CONG NO THANG 10 THANG 11",
    ]

    extractor = W2NERStatementExtractor()
    sep = "=" * 65

    print(f"\n{sep}")
    print("  W2NER DEMO SHOWCASE — Mode:", extractor.mode)
    print(sep)

    for i, text in enumerate(test_cases, 1):
        result = extractor.extract_entities(text)
        print(f"\n[Test {i}] {text!r}")
        print(f"  Tokens      : {result.tokens}")
        print(f"  PAYER_ORG   : {result.payer_orgs}")
        print(f"  SENDER_PER  : {result.sender_persons}")
        print(f"  INVOICE_ID  : {result.invoice_ids}")
        print(f"  DISCOUNT    : {result.discount_rates}")
        print(f"  PERIOD      : {result.periods}")
        print(f"  Spans ({len(result.entity_spans):2d})  : ", end="")
        for sp in result.entity_spans:
            print(
                f"[{sp['label']} | '{sp['text']}' "
                f"| disc={sp.get('is_discontinuous', False)}]",
                end=" "
            )
        print()
        print(f"  Grid rels   : {len(result.grid_2d_Relations)} arcs "
              f"({sum(1 for r in result.grid_2d_Relations if r['relation_type'].startswith('THW'))} THW, "
              f"{sum(1 for r in result.grid_2d_Relations if r['relation_type'] == LABEL_NNW)} NNW)")

    print(f"\n{sep}\n")


# ===========================================================================
# ENTRYPOINT
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    demo_showcase()

