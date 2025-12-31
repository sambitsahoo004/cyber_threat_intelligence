#!/usr/bin/env python3
import json
import re
import argparse
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from datetime import datetime
import random
import sys

# -------------------------
# Configuration / Globals
# -------------------------
RANDOM_SEED = 42

# Safe spaCy loading with fallback
nlp = None
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_trf")
        print("✓ Loaded en_core_web_trf (transformer model)")
    except Exception:
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✓ Loaded en_core_web_sm (small model)")
        except Exception:
            print("ERROR: No spaCy model found!")
            print("Install with: python -m spacy download en_core_web_trf")
            print("Or fallback: python -m spacy download en_core_web_sm")
            sys.exit(1)
except ImportError:
    print("ERROR: spaCy not installed!")
    print("Install with: pip install spacy")
    sys.exit(1)

# -------------------------
# Data classes
# -------------------------
@dataclass
class Mention:
    entity_id: str
    canonical: str
    type: str
    start: int
    end: int
    text: str

@dataclass
class Entity:
    text: str
    type: str
    start: int
    end: int
    canonical: str
    entity_id: str
    confidence: float = 0.9

@dataclass
class Relation:
    source_id: str
    target_id: str
    relation_type: str
    evidence_span: str
    evidence_sentence: str
    certainty: str
    directionality: str
    confidence: float
    negation: bool = False
    metadata: Dict = None
    auto_mapped: bool = False

# -------------------------
# Normalizers
# -------------------------
class IOCNormalizer:
    @staticmethod
    def normalize_cve(cve: str) -> str:
        cve = cve.upper().strip()
        match = re.match(r'CVE[- ]?(\d{4})[- ]?(\d+)', cve, re.IGNORECASE)
        if match:
            return f"CVE-{match.group(1)}-{match.group(2)}"
        return cve

    @staticmethod
    def normalize_hash(hash_str: str) -> str:
        return hash_str.lower().strip().replace(" ", "")

    @staticmethod
    def normalize_ip(ip: str) -> str:
        return ip.strip().rstrip('.')

    @staticmethod
    def normalize_domain(domain: str) -> str:
        return domain.lower().strip().rstrip('.,')

    @staticmethod
    def normalize_url(url: str) -> str:
        return url.strip().rstrip(',')

    @staticmethod
    def normalize(text: str, entity_type: str) -> str:
        if entity_type == "Vulnerability":
            return IOCNormalizer.normalize_cve(text)
        elif "Hash" in entity_type or entity_type == "File":
            return IOCNormalizer.normalize_hash(text)
        elif entity_type == "IP":
            return IOCNormalizer.normalize_ip(text)
        elif entity_type == "Domain":
            return IOCNormalizer.normalize_domain(text)
        elif entity_type == "URL":
            return IOCNormalizer.normalize_url(text)
        else:
            return text.strip()

# -------------------------
# Relationship schema
# (kept as provided)
# -------------------------
class RelationshipSchema:
    RELATIONS = {
        'exploits': {
            'source': ['ThreatActor', 'Malware'],
            'target': ['Vulnerability'],
            'patterns': ['exploit', 'exploited', 'exploiting', 'exploits', 'leverage', 'leveraged', 'abuse', 'abused'],
            'directionality': 'directional'
        },
        'affects': {
            'source': ['Vulnerability'],
            'target': ['Platform', 'Device', 'Software', 'Vendor'],
            'patterns': ['affect', 'affects', 'affected', 'impact', 'impacts', 'impacted'],
            'directionality': 'directional'
        },
        'targets': {
            'source': ['ThreatActor', 'Malware'],
            'target': ['Device', 'Platform', 'Vendor', 'Software'],
            'patterns': ['target', 'targets', 'targeted', 'attack', 'attacks', 'attacked', 'compromise'],
            'directionality': 'directional'
        },
        'communicates_with': {
            'source': ['IP', 'Domain', 'URL', 'Malware', 'File'],
            'target': ['IP', 'Domain', 'URL'],
            'patterns': ['communicate', 'connect', 'beacon', 'callback', 'c2', 'command and control'],
            'directionality': 'symmetric'
        },
        'delivers': {
            'source': ['URL', 'Domain', 'Malware', 'IP'],
            'target': ['Malware', 'File'],
            'patterns': ['deliver', 'drop', 'download', 'distribute', 'host'],
            'directionality': 'directional'
        },
        'belongs_to': {
            'source': ['File', 'Malware', 'IP', 'Domain'],
            'target': ['Malware', 'ThreatActor'],
            'patterns': ['belong', 'associated with', 'linked to', 'part of', 'variant of'],
            'directionality': 'directional'
        },
        'uses': {
            'source': ['ThreatActor', 'Malware'],
            'target': ['Malware', 'Type', 'Function', 'File'],
            'patterns': ['use', 'uses', 'used', 'employ', 'utilize', 'deploy'],
            'directionality': 'directional'
        },
        'includes': {
            'source': ['Malware', 'File'],
            'target': ['File', 'Function'],
            'patterns': ['include', 'contain', 'embed', 'bundle'],
            'directionality': 'directional'
        },
        'evolves_from': {
            'source': ['Vulnerability', 'Malware'],
            'target': ['Vulnerability', 'Malware'],
            'patterns': ['evolve', 'derived from', 'based on', 'successor'],
            'directionality': 'directional'
        },
        'related_to': {
            'source': ['*'],
            'target': ['*'],
            'patterns': ['related to', 'associated with', 'connected to', 'involves'],
            'directionality': 'symmetric'
        }
    }

    HEDGING_WORDS = ['likely', 'possibly', 'probably', 'may', 'might', 'could', 'suspected', 'believed', 'allegedly']
    NEGATION_WORDS = ['not', 'no', 'never', 'neither', 'nor', 'without', 'unable', 'failed', "doesn't", "don't"]

    @classmethod
    def is_valid_relation(cls, source_type: str, target_type: str, relation: str) -> bool:
        if relation not in cls.RELATIONS:
            return False
        schema = cls.RELATIONS[relation]
        if '*' in schema['source'] or '*' in schema['target']:
            return True
        return (source_type in schema['source'] and target_type in schema['target'])

    @classmethod
    def get_directionality(cls, relation: str) -> str:
        return cls.RELATIONS.get(relation, {}).get('directionality', 'directional')

# -------------------------
# Predicate miner
# -------------------------
class PredicateMiner:
    """Mines verb patterns from corpus for relation expansion"""
    def __init__(self):
        self.verb_patterns = Counter()
        self.verb_contexts = defaultdict(list)

    def mine_patterns(self, sent_doc, entities_in_sent):
        """Extract verb patterns between entity pairs using token.i positions"""
        # entities_in_sent are Mention objects with char offsets relative to full doc text
        # Use token.i distances (safer) and sentence boundaries
        try:
            # gather token spans' char ranges
            sent_start_char = sent_doc.start_char
            for token in sent_doc:
                if token.pos_ in ['VERB', 'AUX']:
                    lemma = token.lemma_.lower()
                    # token.i is position in doc; to check left/right, compare character positions
                    token_char_index = token.idx
                    left_ents = [e for e in entities_in_sent if e.end <= token_char_index]
                    right_ents = [e for e in entities_in_sent if e.start >= token_char_index + len(token.text)]
                    if left_ents and right_ents:
                        context = f"{token.text} ({left_ents[0].type} -> {right_ents[0].type})"
                        self.verb_patterns[lemma] += 1
                        self.verb_contexts[lemma].append(context)
        except Exception:
            pass

    def get_suggestions(self, min_freq=3, top_n=15):
        suggestions = []
        for verb, count in self.verb_patterns.most_common(top_n * 2):
            if count >= min_freq:
                suggestions.append({
                    'verb': verb,
                    'frequency': count,
                    'contexts': self.verb_contexts[verb][:5]
                })
        return suggestions[:top_n]

# -------------------------
# Main extractor
# -------------------------
class RelationExtractor:
    def __init__(self, min_confidence=0.45, min_dep_score=0.5, max_related_to_ratio=0.05,
                 sentence_window=1, predicate_min_freq=3, predicate_top_k=15, seed=RANDOM_SEED):
        self.normalizer = IOCNormalizer()
        self.schema = RelationshipSchema()
        self.predicate_miner = PredicateMiner()
        self.entity_id_counter = 0
        self.entities_map: Dict[str, Entity] = {}
        self.min_confidence = float(min_confidence)
        self.min_dep_score = float(min_dep_score)
        self.max_related_to_ratio = float(max_related_to_ratio)
        self.sentence_window = int(sentence_window)
        self.predicate_min_freq = int(predicate_min_freq)
        self.predicate_top_k = int(predicate_top_k)
        self.seed = int(seed)
        self.rng = random.Random(self.seed)  # deterministic RNG
        self.auto_mapped_predicates: Dict[str, int] = {}

    def generate_entity_id(self) -> str:
        self.entity_id_counter += 1
        return f"E{self.entity_id_counter:06d}"

    def robust_mention_finding(self, text: str, entity_text: str, entity_type: str, doc) -> List[Tuple[int, int]]:
        mentions = []

        # 1. Exact text match
        try:
            for m in re.finditer(re.escape(entity_text), text, flags=re.IGNORECASE):
                mentions.append((m.start(), m.end()))
        except re.error:
            pass

        # 2. Canonical form match
        canonical = self.normalizer.normalize(entity_text, entity_type)
        if canonical and canonical != entity_text:
            try:
                for m in re.finditer(re.escape(canonical), text, flags=re.IGNORECASE):
                    mentions.append((m.start(), m.end()))
            except re.error:
                pass

        # 3. Token-based fallback (for partial matches)
        if not mentions and len(entity_text.split()) > 1:
            for token in entity_text.split():
                if len(token) > 3:  # Skip short tokens
                    try:
                        for m in re.finditer(re.escape(token), text, flags=re.IGNORECASE):
                            mentions.append((m.start(), m.end()))
                    except re.error:
                        pass

        # Deduplicate overlapping mentions - proper interval overlap check
        mentions = sorted(set(mentions))
        deduplicated = []
        for start, end in mentions:
            if end <= start:
                continue
            overlaps = False
            for s, e in deduplicated:
                if not (end <= s or start >= e):  # overlap exists
                    overlaps = True
                    break
            if not overlaps:
                deduplicated.append((start, end))

        return deduplicated

    def extract_entities_from_ioc_data(self, ioc_data: Dict) -> Tuple[List[Entity], List[Mention]]:
        entities: List[Entity] = []
        mentions: List[Mention] = []
        text = ioc_data.get('text', '') or ''
        extracted_entities = ioc_data.get('entities', {}) or {}
        confidence_scores = ioc_data.get('confidence_scores', {}) or {}
        doc = nlp(text)  # spaCy doc

        for entity_type, entity_list in extracted_entities.items():
            if not entity_list or entity_list == ['NULL']:
                continue
            confidence = confidence_scores.get(entity_type, 0.9)
            for entity_text in entity_list:
                if not entity_text or entity_text == 'NULL':
                    continue
                canonical = self.normalizer.normalize(entity_text, entity_type)
                key = f"{entity_type}:{canonical}"
                if key not in self.entities_map:
                    entity_id = self.generate_entity_id()
                    ent = Entity(
                        text=entity_text,
                        type=entity_type,
                        start=-1,
                        end=-1,
                        canonical=canonical,
                        entity_id=entity_id,
                        confidence=confidence
                    )
                    self.entities_map[key] = ent
                    entities.append(ent)
                else:
                    ent = self.entities_map[key]
                    if ent not in entities:
                        entities.append(ent)

                # Robust mention finding
                for start, end in self.robust_mention_finding(text, entity_text, entity_type, doc):
                    mentions.append(Mention(
                        entity_id=ent.entity_id,
                        canonical=canonical,
                        type=entity_type,
                        start=start,
                        end=end,
                        text=text[start:end]
                    ))

        return entities, mentions

    def compute_dependency_score(self, doc, ent1_span: Tuple[int, int], ent2_span: Tuple[int, int]) -> float:
        try:
            ent1_tokens = [t for t in doc if ent1_span[0] <= t.idx < ent1_span[1]]
            ent2_tokens = [t for t in doc if ent2_span[0] <= t.idx < ent2_span[1]]

            if not ent1_tokens or not ent2_tokens:
                return 0.0

            min_dist = float('inf')
            for t1 in ent1_tokens:
                for t2 in ent2_tokens:
                    dist = abs(t1.i - t2.i)
                    min_dist = min(min_dist, dist)

            # Normalize: closer = higher score; cap denominator to avoid tiny negatives
            return max(0.0, 1.0 - (min_dist / 20.0))
        except Exception:
            return 0.0

    def detect_certainty(self, text: str) -> str:
        lower = (text or "").lower()
        for hedge in self.schema.HEDGING_WORDS:
            if hedge in lower:
                return 'probable' if hedge in ['likely', 'probably'] else 'possible'
        return 'confirmed'

    def detect_negation(self, text: str) -> bool:
        lower = (text or "").lower()
        return any(neg in lower for neg in self.schema.NEGATION_WORDS)

    def extract_relations_pattern_based(self, text: str, entities: List[Entity], mentions: List[Mention]) -> List[Relation]:
        relations: List[Relation] = []
        doc = nlp(text)
        sentences = list(doc.sents)

        # map mentions to sentence ids
        sent_mentions: Dict[int, List[Mention]] = defaultdict(list)
        for mention in mentions:
            for sid, sent in enumerate(sentences):
                if mention.start >= sent.start_char and mention.end <= sent.end_char:
                    sent_mentions[sid].append(mention)
                    # mine predicates using sentence doc and mentions mapped to that sentence
                    self.predicate_miner.mine_patterns(sent, sent_mentions[sid])
                    break

        seen_relations: Set[Tuple] = set()
        related_to_candidates: List[Dict] = []

        for sid in range(len(sentences)):
            # build window of sentences
            window_sids = [sid] + [s for s in range(sid+1, min(sid + 1 + self.sentence_window, len(sentences)))]
            # collect mentions in window
            window_mentions: List[Mention] = []
            for ws in window_sids:
                window_mentions.extend(sent_mentions.get(ws, []))

            # pairwise combinations (unordered)
            for i in range(len(window_mentions)):
                for j in range(i+1, len(window_mentions)):
                    m1 = window_mentions[i]
                    m2 = window_mentions[j]
                    ent1 = next((e for e in entities if e.entity_id == m1.entity_id), None)
                    ent2 = next((e for e in entities if e.entity_id == m2.entity_id), None)
                    if ent1 is None or ent2 is None:
                        continue

                    window_start = min([sentences[w].start_char for w in window_sids])
                    window_end = max([sentences[w].end_char for w in window_sids])
                    evidence_span = text[window_start:window_end]
                    evidence_span_lower = (evidence_span or "").lower()
                    evidence_sentence = sentences[window_sids[0]].text if window_sids else ""

                    matched_relation = None
                    matched_pattern = None
                    confidence = 0.5

                    # explicit pattern matching (skip 'related_to' initially)
                    for rel_name, rel_info in self.schema.RELATIONS.items():
                        if rel_name == 'related_to':
                            continue
                        for pattern in rel_info.get('patterns', []):
                            if re.search(r'\b' + re.escape(pattern.lower()) + r'\b', evidence_span_lower):
                                # check type constraints (allow reversed if valid)
                                if self.schema.is_valid_relation(ent1.type, ent2.type, rel_name):
                                    matched_relation = rel_name
                                    matched_pattern = pattern
                                    confidence = 0.8
                                    break
                                elif self.schema.is_valid_relation(ent2.type, ent1.type, rel_name):
                                    matched_relation = rel_name
                                    matched_pattern = pattern
                                    confidence = 0.8
                                    # swap to preserve source->target semantics
                                    ent1, ent2, m1, m2 = ent2, ent1, m2, m1
                                    break
                        if matched_relation:
                            break

                    # If no explicit relation, compute dep score and consider for related_to
                    if not matched_relation:
                        dep_score = self.compute_dependency_score(doc, (m1.start, m1.end), (m2.start, m2.end))
                        if dep_score >= self.min_dep_score:
                            related_to_candidates.append({
                                'ent1': ent1, 'ent2': ent2,
                                'm1': m1, 'm2': m2,
                                'evidence_span': evidence_span,
                                'evidence_sentence': evidence_sentence,
                                'confidence': dep_score,
                                'window_start': window_start
                            })
                        continue

                    # Emit matched relation after checks
                    certainty = self.detect_certainty(evidence_span)
                    negation = self.detect_negation(evidence_span)

                    if confidence < self.min_confidence:
                        continue

                    dkey = (ent1.entity_id, ent2.entity_id, matched_relation, window_start)
                    if dkey in seen_relations:
                        continue
                    seen_relations.add(dkey)

                    rel = Relation(
                        source_id=ent1.entity_id,
                        target_id=ent2.entity_id,
                        relation_type=matched_relation,
                        evidence_span=(evidence_span or "")[:500],
                        evidence_sentence=evidence_sentence,
                        certainty=certainty,
                        directionality=self.schema.get_directionality(matched_relation),
                        confidence=confidence,
                        negation=negation,
                        metadata={
                            'source_entity': ent1.canonical,
                            'source_type': ent1.type,
                            'target_entity': ent2.canonical,
                            'target_type': ent2.type,
                            'pattern_matched': matched_pattern
                        },
                        auto_mapped=False
                    )
                    relations.append(rel)

        # Cap related_to ratio based on explicit relations already emitted
        num_explicit = len(relations)
        if self.max_related_to_ratio <= 0 or num_explicit == 0:
            allowed_related = 0
        else:
            allowed_related = int((num_explicit * self.max_related_to_ratio) / (1.0 - self.max_related_to_ratio))
            allowed_related = max(0, allowed_related)

        # deterministically sample related_to candidates
        if len(related_to_candidates) > allowed_related:
            related_to_candidates = self.rng.sample(related_to_candidates, allowed_related)

        # Emit related_to relations
        for cand in related_to_candidates:
            dkey = (cand['ent1'].entity_id, cand['ent2'].entity_id, 'related_to', cand['window_start'])
            if dkey not in seen_relations:
                seen_relations.add(dkey)
                rel = Relation(
                    source_id=cand['ent1'].entity_id,
                    target_id=cand['ent2'].entity_id,
                    relation_type='related_to',
                    evidence_span=(cand['evidence_span'] or "")[:500],
                    evidence_sentence=cand['evidence_sentence'],
                    certainty='possible',
                    directionality='symmetric',
                    confidence=cand['confidence'],
                    negation=False,
                    metadata={
                        'source_entity': cand['ent1'].canonical,
                        'source_type': cand['ent1'].type,
                        'target_entity': cand['ent2'].canonical,
                        'target_type': cand['ent2'].type,
                        'pattern_matched': 'dependency_only'
                    },
                    auto_mapped=False
                )
                relations.append(rel)

        return relations

    def process_ioc_document(self, ioc_data: Dict) -> Dict:
        entities, mentions = self.extract_entities_from_ioc_data(ioc_data)
        text = ioc_data.get('text', '') or ''
        relations = self.extract_relations_pattern_based(text, entities, mentions)

        return {
            'document_id': ioc_data.get('source_id', 'unknown'),
            'source': ioc_data.get('source', 'unknown'),
            'created_date': ioc_data.get('created_date', datetime.now().isoformat()),
            'entities': [asdict(e) for e in entities],
            'mentions': [asdict(m) for m in mentions],
            'relations': [asdict(r) for r in relations],
            'statistics': {
                'total_entities': len(entities),
                'total_mentions': len(mentions),
                'total_relations': len(relations),
                'relation_types': dict(Counter(r['relation_type'] for r in [asdict(rr) for rr in relations]))
            }
        }

    def save_triples_jsonl(self, processed_docs: List[Dict], output_file: str):
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in processed_docs:
                entities_map = {e['entity_id']: e for e in doc['entities']}
                for rel in doc['relations']:
                    head = entities_map.get(rel['source_id'], {})
                    tail = entities_map.get(rel['target_id'], {})
                    triple = {
                        'head_id': rel['source_id'],
                        'head_text': head.get('canonical', ''),
                        'head_type': head.get('type', ''),
                        'tail_id': rel['target_id'],
                        'tail_text': tail.get('canonical', ''),
                        'tail_type': tail.get('type', ''),
                        'relation': rel['relation_type'],
                        'confidence': rel.get('confidence', 0.0),
                        'certainty': rel.get('certainty', ''),
                        'negation': rel.get('negation', False),
                        'evidence_span': rel.get('evidence_span', ''),
                        'sentence': rel.get('evidence_sentence', ''),
                        'source_id': doc.get('document_id', ''),
                        'source': doc.get('source', ''),
                        'auto_mapped': rel.get('auto_mapped', False)
                    }
                    f.write(json.dumps(triple, ensure_ascii=False) + '\n')
        print(f"✓ Saved triples to {output_file}")

# -------------------------
# CLI main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description='Robust IOC Relation Extractor')
    parser.add_argument('--input', default='balanced_ioc_dataset.json', help='Input JSON file (single doc or list)')
    parser.add_argument('--min_confidence', type=float, default=0.45, help='Minimum confidence threshold')
    parser.add_argument('--min_dep_score', type=float, default=0.5, help='Minimum dependency score')
    parser.add_argument('--max_related_to_ratio', type=float, default=0.05, help='Max related_to ratio (e.g., 0.05 = 5%%)')
    parser.add_argument('--sentence_window', type=int, default=1, help='Sentence window size for relation extraction')
    parser.add_argument('--predicate_min_freq', type=int, default=3, help='Min frequency for predicate suggestions')
    parser.add_argument('--predicate_top_k', type=int, default=15, help='Top K predicate suggestions')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed for reproducibility')
    parser.add_argument('--reset_entities', action='store_true', help='Reset entity registry per document')
    parser.add_argument('--out', default='triples_output.jsonl', help='Triples JSONL output path')
    args = parser.parse_args()

    # set deterministic RNG
    random.seed(int(args.seed))

    print(f"Loading data from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    docs = [raw] if isinstance(raw, dict) else raw
    print(f"Loaded {len(docs)} documents")

    extractor = RelationExtractor(
        min_confidence=args.min_confidence,
        min_dep_score=args.min_dep_score,
        max_related_to_ratio=args.max_related_to_ratio,
        sentence_window=args.sentence_window,
        predicate_min_freq=args.predicate_min_freq,
        predicate_top_k=args.predicate_top_k,
        seed=args.seed
    )

    all_results = []
    for idx, doc in enumerate(docs):
        print(f"Processing document {idx + 1}/{len(docs)}...")
        if args.reset_entities:
            extractor.entity_id_counter = 0
            extractor.entities_map = {}
        result = extractor.process_ioc_document(doc)
        all_results.append(result)

    # Save processed docs and triples
    with open('processed_ioc_relations.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("✓ Saved processed_ioc_relations.json")

    extractor.save_triples_jsonl(all_results, args.out)

    # predicate suggestions
    suggestions = extractor.predicate_miner.get_suggestions(min_freq=args.predicate_min_freq, top_n=args.predicate_top_k)
    with open('predicate_suggestions.json', 'w', encoding='utf-8') as f:
        json.dump(suggestions, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(suggestions)} predicate suggestions")

    # statistics
    total_relations = sum(len(doc['relations']) for doc in all_results)
    related_to_count = sum(1 for doc in all_results for rel in doc['relations'] if rel['relation_type'] == 'related_to')
    pct = (related_to_count / total_relations * 100.0) if total_relations > 0 else 0.0
    print("\n" + "="*50)
    print(f"Total relations: {total_relations}")
    print(f"Related_to: {related_to_count} ({pct:.1f}%)")
    print("="*50)

if __name__ == "__main__":
    main()
